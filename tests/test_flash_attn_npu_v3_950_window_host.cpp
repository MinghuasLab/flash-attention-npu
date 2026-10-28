// Compare the production metadata normalization with the API's bottom-right
// window definition, including padded bounds and shorter per-batch lengths.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "fa_metadata_args.h"

int main()
{
    uint64_t comparisons = 0;
    for (int64_t nq : {1, 3, 8, 17, 65, 257}) {
        for (int64_t nk : {1, 4, 9, 16, 65}) {
            for (bool padded : {false, true}) {
                const int64_t qb = nq + (padded ? 7 : 0);
                const int64_t kb = padded ? (nk + 127) / 128 * 128 : nk;
                for (bool causal : {false, true}) {
                    for (int64_t wl : {-1L, 0L, 1L, nk, kb}) {
                        for (int64_t wr : {-1L, 0L, 2L, nk, nq, qb}) {
                            const auto mask = DeriveFwdMask(causal, wl, wr, qb, kb);
                            assert(mask.maskType == (mask.is_local ? 4U : (mask.is_causal ? 1U : 0U)));
                            for (int64_t i = 0; i < nq; ++i) {
                                for (int64_t j = 0; j < nk; ++j) {
                                    const int64_t center = i + nk - nq;
                                    const int64_t right = causal ? 0 : wr;
                                    const bool expected = (wl < 0 || j >= center - wl) &&
                                                          (right < 0 || j <= center + right);
                                    bool actual = true;
                                    if (mask.is_causal) {
                                        actual = j <= center;
                                    } else if (mask.is_local) {
                                        actual = j >= center - std::min(mask.window_left, nk) &&
                                                 j <= center + std::min(mask.window_right, nq);
                                    }
                                    assert(actual == expected);
                                    ++comparisons;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << comparisons << " mask decisions matched\n";
}
