"""Compile separable integer mask expressions into tile-local TLA helpers.

Only the BF16 paired attention kernel uses this lowering. It hoists auxiliary
reads that depend on one token axis, then evaluates the remaining predicate in
SIMD. Unsupported callbacks keep the existing SIMT implementation. The planner
never reads tensor contents, changes block metadata, or infers identity maps.
Generated helpers and the bounded source cache contain no launch-time tensors.
"""

import ast
import hashlib
import inspect
import operator
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
import catlass.tla as tla

# Reuse a decorator factory. Creating it during lowering captures that call
# stack in the current CATLASS release until cyclic GC runs.
_device_helper = tla.jit()


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple
    dtype: str = "Int32"

    @property
    def deps(self):
        if self.op == "var":
            return frozenset((self.args[0],)) if self.args[0] in ("q", "k") else frozenset()
        return frozenset().union(*(a.deps for a in self.args if isinstance(a, Expr)))

    @property
    def has_load(self):
        return self.op == "load" or any(a.has_load for a in self.args if isinstance(a, Expr))

    def __bool__(self):
        raise NotImplementedError("symbolic truth outside an interpreted if")


def const(v, dtype=None):
    if dtype == "Float32" or isinstance(v, float):
        raise NotImplementedError("floating point mask expression")
    if isinstance(v, Expr):
        return v if dtype is None or v.dtype == dtype else Expr("cast", (v,), dtype)
    if not isinstance(v, (int, float, bool)):
        raise NotImplementedError(type(v))
    if isinstance(v, int) and not isinstance(v, bool) and dtype in (None, "Int32"):
        if not -(2**31) <= v < 2**31:
            raise NotImplementedError("wide implicit constant")
    return Expr(
        "const",
        (v,),
        dtype
        or ("Bool" if isinstance(v, bool) else "Float32" if isinstance(v, float) else "Int32"),
    )


def expr(op, a, b=None):
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        return OPS[op](a, b) if b is not None else OPS[op](a)
    a = const(a)
    if b is None:
        return Expr(op, (a,), "Bool" if op == "not" else a.dtype)
    b = const(b)
    if (a.dtype == "Bool") != (b.dtype == "Bool"):
        raise NotImplementedError("mixed boolean and integer operands")
    if op in ("&", "|"):
        if a.op == "const" and a.dtype == "Bool":
            return (b if a.args[0] else a) if op == "&" else (a if a.args[0] else b)
        if b.op == "const" and b.dtype == "Bool":
            return (a if b.args[0] else b) if op == "&" else (b if b.args[0] else a)
        if a == b:
            return a
    if a.op == "const" and a.dtype != b.dtype:
        a = const(a.args[0], b.dtype)
    if b.op == "const" and b.dtype != a.dtype:
        b = const(b.args[0], a.dtype)
    if a.dtype != b.dtype:
        raise NotImplementedError("mixed integer widths")
    if op in ("//", "%", "<<", ">>") and (b.op != "const" or b.args[0] <= 0):
        raise NotImplementedError("possibly trapping arithmetic")
    dtype = "Bool" if op in ("==", "!=", ">=", "<=", ">", "<") else a.dtype
    return Expr(op, (a, b), dtype)


def select(p, a, b):
    a, b, p = const(a), const(b), const(p)
    if p.dtype != "Bool" or a.dtype != b.dtype:
        raise NotImplementedError("nonboolean condition or mixed branch types")
    if a == b:
        return a
    if b.op == "const" and b.dtype == "Bool" and b.args == (False,):
        return expr("&", p, a)
    if b == p:
        return expr("&", p, a)
    return Expr("select", (p, a, b), a.dtype)


OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "neg": operator.neg,
    "~": operator.invert,
    "not": operator.not_,
}
OPN = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Gt: ">",
    ast.Lt: "<",
    ast.GtE: ">=",
    ast.LtE: "<=",
}


@dataclass
class Aux:
    number: int
    dtype: str
    shape: tuple
    stride: tuple
    unsigned: bool = False

    def get(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        if len(indices) != len(self.shape):
            raise NotImplementedError("auxiliary rank")
        if self.unsigned:
            raise NotImplementedError("packed unsigned values")
        return Expr("load", (self.number, *(const(i) for i in indices)), self.dtype)


class Trace:
    def __init__(self):
        self.active = set()

    def call(self, fn, args, kwargs=None):
        kwargs = kwargs or {}
        if fn is tla.const_expr:
            if isinstance(args[0], Expr):
                raise NotImplementedError("dynamic constexpr")
            return args[0]
        if fn in (tla.Int32, tla.Int64, tla.Bool, tla.Float32):
            return const(args[0], fn.__name__)
        if getattr(fn, "__name__", None) == "_views" and getattr(fn, "__module__", "").endswith(
            ".auxiliary"
        ):
            return args[0]
        if getattr(fn, "__name__", None) == "_metadata_size" and getattr(
            fn, "__module__", ""
        ).endswith(".auxiliary"):
            return fn(*args)
        if fn is len:
            return len(args[0])
        if isinstance(fn, tuple) and fn[0] == "cast":
            return const(fn[1], args[0].__name__)
        fn = inspect.unwrap(getattr(fn, "fn", fn))
        if not inspect.isfunction(fn) or id(fn) in self.active:
            raise NotImplementedError("unsupported helper")
        self.active.add(id(fn))
        try:
            code = ast.parse(inspect.cleandoc(inspect.getsource(fn)))
            definition = next(
                n for n in code.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            env = dict(fn.__globals__)
            env.update(inspect.getclosurevars(fn).nonlocals)
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
            env.update(bound.arguments)
            result, returned = self.block(definition.body, env)
            if not returned:
                raise NotImplementedError("missing return")
            return result
        finally:
            self.active.remove(id(fn))

    def value(self, n, env):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, (ast.Tuple, ast.List)):
            return tuple(self.value(x, env) for x in n.elts)
        if isinstance(n, ast.Attribute):
            obj = self.value(n.value, env)
            if isinstance(obj, Expr) and n.attr == "to":
                return ("cast", obj)
            return getattr(obj, n.attr)
        if isinstance(n, ast.Slice):
            return slice(
                *(self.value(x, env) if x is not None else None for x in (n.lower, n.upper, n.step))
            )
        if isinstance(n, ast.Subscript):
            obj, idx = self.value(n.value, env), self.value(n.slice, env)
            return obj.get(idx) if isinstance(obj, Aux) else obj[idx]
        if isinstance(n, ast.BinOp):
            return expr(OPN[type(n.op)], self.value(n.left, env), self.value(n.right, env))
        if isinstance(n, ast.UnaryOp):
            op = {ast.Not: "not", ast.Invert: "~", ast.USub: "neg"}[type(n.op)]
            return expr(op, self.value(n.operand, env))
        if isinstance(n, ast.Compare):
            left = self.value(n.left, env)
            result = True
            for op, right_ast in zip(n.ops, n.comparators):
                right = self.value(right_ast, env)
                if isinstance(op, ast.Is):
                    one = left is right
                elif isinstance(op, ast.IsNot):
                    one = left is not right
                else:
                    one = expr(OPN[type(op)], left, right)
                result = expr("&", result, one)
                left = right
            return result
        if isinstance(n, ast.BoolOp):
            value = self.value(n.values[0], env)
            if not (isinstance(value, bool) or isinstance(value, Expr) and value.dtype == "Bool"):
                raise NotImplementedError("nonboolean logical operand")
            for other in n.values[1:]:
                if not isinstance(value, Expr):
                    if isinstance(n.op, ast.And) and not value:
                        return value
                    if isinstance(n.op, ast.Or) and value:
                        return value
                right = self.value(other, env)
                if not (
                    isinstance(right, bool) or isinstance(right, Expr) and right.dtype == "Bool"
                ):
                    raise NotImplementedError("nonboolean logical operand")
                value = expr("&" if isinstance(n.op, ast.And) else "|", value, right)
            return value
        if isinstance(n, ast.Call):
            return self.call(
                self.value(n.func, env),
                tuple(self.value(x, env) for x in n.args),
                {k.arg: self.value(k.value, env) for k in n.keywords},
            )
        if isinstance(n, ast.IfExp):
            p = self.value(n.test, env)
            return (
                select(p, self.value(n.body, env), self.value(n.orelse, env))
                if isinstance(p, Expr)
                else self.value(n.body if p else n.orelse, env)
            )
        raise NotImplementedError(type(n).__name__)

    def block(self, body, env):
        for n in body:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                continue
            if isinstance(n, ast.Return):
                return self.value(n.value, env), True
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                env[n.targets[0].id] = self.value(n.value, env)
            elif isinstance(n, ast.If):
                p = self.value(n.test, env)
                if not isinstance(p, Expr):
                    result, done = self.block(n.body if p else n.orelse, env)
                    if done:
                        return result, True
                else:
                    yes, no = env.copy(), env.copy()
                    y, yr = self.block(n.body, yes)
                    z, zr = self.block(n.orelse, no)
                    if yr or zr:
                        if yr and zr:
                            return select(p, y, z), True
                        raise NotImplementedError("conditional early return")
                    for key in set(yes) | set(no):
                        if key not in yes or key not in no:
                            continue
                        if yes[key] is not no[key]:
                            env[key] = select(p, yes[key], no[key])
            else:
                raise NotImplementedError(type(n).__name__)
        return None, False


def specs_of(function):
    seen = set()

    def visit(fn):
        fn = inspect.unwrap(getattr(fn, "fn", fn))
        if not inspect.isfunction(fn) or id(fn) in seen:
            return ()
        seen.add(id(fn))
        values = inspect.getclosurevars(fn).nonlocals
        if "specs" in values:
            return values["specs"]
        for value in values.values():
            result = visit(value)
            if result:
                return result
        return ()

    return visit(function)


class Emit:
    def __init__(self, aux, lines, indent="    ", cuts=None, axis=None):
        self.aux, self.lines, self.indent, self.cuts, self.axis = (
            aux,
            lines,
            indent,
            cuts or {},
            axis,
        )
        self.memo = {}
        self.prefix = "v"

    def put(self, line):
        self.lines.append(self.indent + line)

    def get(self, e):
        if not isinstance(e, Expr):
            return repr(e)
        if e in self.memo:
            return self.memo[e]
        if e in self.cuts:
            offset, width, axis = self.cuts[e]
            index = "qi" if axis == "q" else "ki"
            v = f"cache[{offset} + {index} * {width}]"
            if width == 2:
                v = f"(({v}.to(tla.Int64) & tla.Int64(0xFFFFFFFF)) | (cache[{offset + 1} + {index} * 2].to(tla.Int64) << tla.Int64(32)))"
            elif e.dtype == "Bool":
                v = f"({v} != 0)"
        elif e.op == "const":
            v = f"tla.{e.dtype}({e.args[0]!r})"
        elif e.op == "var":
            v = e.args[0]
        elif e.op == "cast":
            v = f"{self.get(e.args[0])}.to(tla.{e.dtype})"
        elif e.op == "load":
            a = self.aux[e.args[0]]
            coords = [self.get(i) for i in e.args[1:]]
            bounds = [
                f"({i} >= tla.{node.dtype}(0)) & ({i} < {self.get(size)}.to(tla.{node.dtype}))"
                for i, node, size in zip(coords, e.args[1:], a.shape)
            ]
            offset = " + ".join(
                f"({i}.to(tla.Int64) * {self.get(st)}.to(tla.Int64))"
                for i, st in zip(coords, a.stride)
            )
            name = self.prefix + str(len(self.memo))
            self.memo[e] = name
            self.put(f"{name} = tla.{e.dtype}(0)")
            self.put("if " + " & ".join(f"({b})" for b in bounds) + ":")
            self.put(f"    {name} = aux[{a.number}][{offset}]")
            return name
        elif e.op == "select":
            cond, a, b = (self.get(i) for i in e.args)
            name = self.prefix + str(len(self.memo))
            self.memo[e] = name
            self.put(f"{name} = {b}")
            self.put(f"if {cond}:")
            self.put(f"    {name} = {a}")
            return name
        elif e.op in ("not", "~", "neg"):
            v = f"({'not ' if e.op == 'not' else '-' if e.op == 'neg' else '~'}{self.get(e.args[0])})"
        else:
            v = f"({self.get(e.args[0])} {e.op} {self.get(e.args[1])})"
        name = self.prefix + str(len(self.memo))
        self.memo[e] = name
        self.put(f"{name} = {v}")
        return name


@dataclass
class Plan:
    stage: object
    evaluate: object
    words: int
    source: str
    vector: object


def build(mask_mod, tensors, scalars):
    try:
        return _build(mask_mod, tensors, scalars)
    except (
        NotImplementedError,
        KeyError,
        TypeError,
        ValueError,
        StopIteration,
        AttributeError,
        OSError,
        SyntaxError,
        IndexError,
        RecursionError,
    ):
        return None


def _build(mask_mod, tensors, scalars):
    specs = specs_of(mask_mod)
    if not specs or len(specs) != len(tensors):
        raise NotImplementedError("auxiliary specs")
    scalar_types = tuple(type(value).__name__ for value in scalars)
    if any(dtype != "Int32" for dtype in scalar_types):
        raise NotImplementedError("non-Int32 scalar or wide auxiliary metadata")
    scalar_nodes = tuple(
        Expr("var", (f"scalars[{i}]",), dtype) for i, dtype in enumerate(scalar_types)
    )
    metadata = sum(2 * len(s[0]) for s in specs if len(s) == 4)
    offset = len(scalars) - metadata
    aux = []
    for i, (s, t) in enumerate(zip(specs, tensors)):
        shape, stride, unsigned = s[:3]
        dtype = t.ptr.dtype.__name__
        if dtype not in ("Int32", "Int64"):
            raise NotImplementedError("noninteger auxiliary")
        if len(s) == 4:
            rank = len(shape)
            shape = scalar_nodes[offset : offset + rank]
            stride = tuple(
                const(v) if v is not None else scalar_nodes[offset + rank + j]
                for j, v in enumerate(stride)
            )
            offset += 2 * rank
        else:
            if any(not 0 <= x < 2**31 for x in (*shape, *stride)):
                raise NotImplementedError("wide auxiliary metadata")
            shape = tuple(const(x) for x in shape)
            stride = tuple(const(x) for x in stride)
        aux.append(Aux(i, dtype, tuple(shape), tuple(stride), unsigned))
    info = tuple(Expr("var", (f"info[{i}]",)) for i in range(4))
    variables = [Expr("var", (v,)) for v in ("b", "h", "q", "k")]
    root = const(Trace().call(mask_mod, (0.0, *variables, tuple(aux), scalar_nodes, info, True)))
    if root.dtype != "Bool":
        raise NotImplementedError("nonboolean mask result")
    root = narrow_guarded_comparisons(normalize(root))
    cuts = {}
    words = 0

    def scan(e):
        nonlocal words
        if not isinstance(e, Expr):
            return
        if e.deps in (frozenset(("q",)), frozenset(("k",))) and e.has_load:
            if e.dtype not in ("Int32", "Int64", "Bool"):
                raise NotImplementedError("noninteger cached expression")
            if e not in cuts:
                axis = next(iter(e.deps))
                width = 2 if e.dtype == "Int64" else 1
                cuts[e] = (words, width, axis)
                words += (64 if axis == "q" else 256) * width
            return
        if e.op == "load":
            raise NotImplementedError("mixed or uniform auxiliary load")
        for a in e.args:
            scan(a)

    scan(root)
    if not cuts or words > 2080:
        raise NotImplementedError(f"scratch budget {words}")
    lines = [
        "import catlass.tla as tla",
        "",
        "@tla.jit",
        "def stage(cache, aux, scalars, qstart, vi, k0, k1, b, h, info):",
        "    tid, _, _ = tla.arch.thread_idx()",
        "    q = tla.Int32(0)",
        "    k = tla.Int32(0)",
    ]
    for ordinal, (axis, start, limit, index) in enumerate(
        [("q", "qstart + vi * 64", 64, "tid"), ("k", "k0", 256, "tid")]
    ):
        base = (0, 64, 192)[ordinal]
        local = "tid" if base == 0 else f"(tid - {base})"
        index = local if ordinal < 2 else f"({local} + 128)"
        lines += [f"    if (tid >= {base}) & (tid < {base + limit}):"]
        if axis == "q":
            lines += [f"        q = {start} + {local}"]
        else:
            lines += [
                "        selected_k = k0",
                "        if tid >= 192:",
                "            selected_k = k1",
                "        k = selected_k + ((tid - 64) & 127)",
            ]
        emit = Emit(aux, lines, "        ")
        emit.prefix = f"s{ordinal}_v"
        loads = []

        def collect_loads(e):
            if not isinstance(e, Expr):
                return
            for a in e.args:
                collect_loads(a)
            if e.op == "load" and e not in loads:
                loads.append(e)

        for node, (_, _, node_axis) in cuts.items():
            if node_axis == axis:
                collect_loads(node)
        for node in loads:
            emit.get(node)
        for e, (offset, width, a) in cuts.items():
            if a != axis:
                continue
            value = emit.get(e)
            lines.append(f"        cache[{offset} + ({index}) * {width}] = {value}.to(tla.Int32)")
            if width == 2:
                lines.append(
                    f"        cache[{offset + 1} + ({index}) * 2] = ({value} >> tla.Int64(32)).to(tla.Int32)"
                )
    lines += ["", "@tla.jit", "def evaluate(cache, qi, ki, b, h, q, k, info, scalars):"]
    emit = Emit(aux, lines, cuts=cuts)
    result = emit.get(root)
    lines.append("    return " + result)
    lines += vector_source(aux, cuts, root)
    source = "\n".join(lines) + "\n"
    stage, evaluate, vector, path = _compile_source(source)
    return Plan(stage, evaluate, words, path, vector)


@lru_cache(maxsize=128)
def _compile_source(source):
    """Keep only generated code in the cache, never callback closure owners.

    Source files let TLA inspect helpers during lowering without unbounded
    additions to sys.modules or linecache. A source hash also makes concurrent
    compilations share the same immutable file.
    """
    source = source.replace("@tla.jit\n", "@_device_helper\n")
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    folder = cache_root / "catlass" / "separable_masks"
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    path = folder / (fingerprint + ".py")
    if not path.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as temporary:
            temporary.write(source)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    namespace = {
        "__name__": "_tla_separable_" + fingerprint,
        "__file__": str(path),
        "_device_helper": _device_helper,
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["stage"], namespace["evaluate"], namespace["vector"], str(path)


class VectorEmit:
    def __init__(self, lines, cuts, indent="    ", memo=None, prefix="w"):
        self.lines, self.cuts, self.indent = lines, cuts, indent
        self.memo = dict(memo or {})
        self.prefix = prefix

    def put(self, s):
        self.lines.append(self.indent + s)

    def name(self, e, source):
        n = self.prefix + str(len(self.memo))
        self.memo[e] = n
        self.put(f"{n} = {source}")
        return n

    def get(self, e):
        if e in self.memo:
            return self.memo[e]
        if e in self.cuts:
            offset, width, axis = self.cuts[e]
            n = self.prefix + str(len(self.memo))
            if width == 2:
                if axis == "q":
                    self.put(
                        f"{n}a = tla.tile_view(cache, tla.make_shape(64), tla.make_coord({offset // 64})).load()"
                    )
                    self.put(
                        f"{n}b = tla.tile_view(cache, tla.make_shape(64), tla.make_coord({offset // 64 + 1})).load()"
                    )
                    self.put(f"{n}lo, {n}hi = tla.deinterleave({n}a, {n}b)")
                else:
                    self.put(
                        f"{n}lo = tla.tile_view(cache, tla.make_shape(1), tla.make_coord({offset} + ki * 2)).load(brc)"
                    )
                    self.put(
                        f"{n}hi = tla.tile_view(cache, tla.make_shape(1), tla.make_coord({offset + 1} + ki * 2)).load(brc)"
                    )
                self.memo[e] = (n + "lo", n + "hi")
                return self.memo[e]
            if axis == "q":
                source = f"tla.tile_view(cache, tla.make_shape(64), tla.make_coord({offset // 64})).load()"
            else:
                source = f"tla.tile_view(cache, tla.make_shape(1), tla.make_coord({offset} + ki)).load(brc)"
            self.put(f"{n} = {source}")
            if e.dtype == "Bool":
                self.put(f'{n} = tla.cmp({n}, 0, mode="ne")')
            self.memo[e] = n
            return n
        if e.op == "load":
            raise NotImplementedError("vector mixed auxiliary")
        if e.op == "const":
            if e.dtype == "Int64":
                raise NotImplementedError("vector wide constant")
            if e.dtype == "Bool":
                return self.name(
                    e,
                    f"tla.create_mask(pattern=tla.mask.{'ALL' if e.args[0] else 'NONE'}, dtype=tla.Int32)",
                )
            return self.name(e, f"tla.full({e.args[0]}, tla.{e.dtype})")
        if e.op == "var":
            if e.args[0] == "q":
                source = "tla.add(tla.arange(0, dtype=tla.Int32), qstart + vi * 64)"
            elif e.args[0] == "k":
                source = "tla.add(tla.full(0, tla.Int32), kbase + key)"
            else:
                source = f"tla.add(tla.full(0, tla.Int32), {e.args[0]})"
            return self.name(e, source)
        if e.op == "cast":
            a = self.get(e.args[0])
            if e.dtype == e.args[0].dtype:
                return a
            raise NotImplementedError("vector cross-axis cast")
        values = [self.get(a) for a in e.args]
        cmp = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
        if any(isinstance(x, tuple) for x in values):
            if len(values) != 2 or e.op not in cmp or not all(isinstance(x, tuple) for x in values):
                raise NotImplementedError("wide cross-axis vector arithmetic")
            a, b = values
            eq = f'tla.bitwise_and(tla.cmp({a[1]}, {b[1]}, mode="eq"), tla.cmp({a[0]}, {b[0]}, mode="eq"))'
            gt = f'tla.bitwise_or(tla.cmp({a[1]}, {b[1]}, mode="gt"), tla.bitwise_and(tla.cmp({a[1]}, {b[1]}, mode="eq"), tla.cmp({a[0]}.bitcast(tla.UInt32), {b[0]}.bitcast(tla.UInt32), mode="gt")))'
            lt = f'tla.bitwise_or(tla.cmp({a[1]}, {b[1]}, mode="lt"), tla.bitwise_and(tla.cmp({a[1]}, {b[1]}, mode="eq"), tla.cmp({a[0]}.bitcast(tla.UInt32), {b[0]}.bitcast(tla.UInt32), mode="lt")))'
            source = {
                "==": eq,
                "!=": f"tla.bitwise_not({eq})",
                ">": gt,
                "<": lt,
                ">=": f"tla.bitwise_not({lt})",
                "<=": f"tla.bitwise_not({gt})",
            }[e.op]
        elif e.op in cmp:
            source = f'tla.cmp({values[0]}, {values[1]}, mode="{cmp[e.op]}")'
        elif e.op == "select":
            source = f"tla.where({values[0]}, {values[1]}, {values[2]})"
        elif e.op in ("&", "|", "^", "+", "-", "*"):
            fn = {
                "&": "bitwise_and",
                "|": "bitwise_or",
                "^": "bitwise_xor",
                "+": "add",
                "-": "sub",
                "*": "mul",
            }[e.op]
            source = f"tla.{fn}({values[0]}, {values[1]})"
        else:
            raise NotImplementedError("vector op " + e.op)
        return self.name(e, source)


def vector_source(aux, cuts, root):
    lines = [
        "",
        "@tla.jit",
        "def vector(cache, mask0, mask1, qstart, vi, k0, k1, b, h, info, scalars, second: tla.Constexpr=True):",
        "    brc = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)",
        "    store_mask = tla.update_mask(8, dtype=tla.Int32)[0]",
    ]
    setup = VectorEmit(lines, cuts, prefix="qv")
    for e, (_, _, axis) in cuts.items():
        if axis == "q":
            setup.get(e)
    lines += [
        "    for part in tla.range_constexpr(2):",
        "        if tla.const_expr(part == 0 or second):",
        "            target = mask0 if tla.const_expr(part == 0) else mask1",
        "            kbase = k0 if tla.const_expr(part == 0) else k1",
        "            for key in tla.range(128):",
        "                ki = key + part * 128",
    ]
    emit = VectorEmit(lines, cuts, "                ", setup.memo, prefix="kv")
    pred = emit.get(root)
    emit.put(
        f"tla.tile_view(target, tla.make_shape(1, 8), tla.make_coord(key, 0)).store({pred}, tla.params.MaskStoreParams())"
    )
    lines += ["    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)"]
    return lines


def normalize(e):
    if not isinstance(e, Expr):
        return e
    e = Expr(e.op, tuple(normalize(a) for a in e.args), e.dtype)
    if e.op not in ("&", "|") or e.dtype != "Bool":
        return e
    terms = []

    def flatten(a):
        if isinstance(a, Expr) and a.op == e.op and a.dtype == "Bool":
            for x in a.args:
                flatten(x)
        else:
            terms.append(a)

    for a in e.args:
        flatten(a)
    groups = {}
    for a in terms:
        deps = a.deps
        groups.setdefault(deps, [])
        if a not in groups[deps]:
            groups[deps].append(a)
    result = None
    for values in groups.values():
        part = values[0]
        for a in values[1:]:
            part = expr(e.op, part, a)
        result = part if result is None else expr(e.op, result, part)
    return result


def narrow_guarded_comparisons(root):
    """Use Int32 comparisons only under independent proof of both bounds.

    Top-level AND guards are preserved with their original Int64 semantics.
    Only comparisons mixing q and k are narrowed, so values rejected by any
    guard cannot turn a false final mask into a true mask.
    """
    conjuncts = []

    def flatten(e):
        if e.op == "&" and e.dtype == "Bool":
            for a in e.args:
                flatten(a)
        else:
            conjuncts.append(e)

    flatten(root)
    lower, upper = set(), set()
    for e in conjuncts:
        if e.op not in (">=", ">", "<", "<="):
            continue
        a, b = e.args
        if a.dtype != "Int64":
            continue
        if e.op in (">=", ">") and b.op == "const" and b.args[0] >= 0:
            lower.add(a)
        if e.op in ("<", "<=") and (
            (b.op == "cast" and b.args[0].dtype == "Int32")
            or (b.op == "const" and b.args[0] <= 2**31 - 1)
        ):
            upper.add(a)
    bounded = lower & upper

    def visit(e):
        if not isinstance(e, Expr):
            return e
        if e.op in ("==", "!=", "<", "<=", ">", ">=") and e.deps == frozenset(("q", "k")):
            a, b = e.args
            if a in bounded and b in bounded:
                return Expr(e.op, (const(a, "Int32"), const(b, "Int32")), "Bool")
        return Expr(e.op, tuple(visit(a) for a in e.args), e.dtype)

    return visit(root)
