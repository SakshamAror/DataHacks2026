## Expression

```
best_bid_size = yes_book.bids[0].size
best_ask_size = yes_book.asks[0].size

feat_yes_obi_level1 = (best_bid_size − best_ask_size) / (best_bid_size + best_ask_size)
```

Best-level Order Book Imbalance (OBI) on the YES side of the Polymarket order book.
Range: `[−1, 1]`.

---

## Motivation

The level-1 OBI is the **most widely validated short-horizon microstructure predictor**
in academic literature. The intuition is straightforward:

- If there are 500 YES shares resting on the bid and only 100 on the ask, buying interest
  far outweighs selling interest at the inside market. The next price move is more likely
  to be upward (YES price rising → higher P(YES) at settlement).

- Conversely, a heavy ask side signals that YES sellers are positioning at the current
  price level — likely anticipating a downward move.

**Mathematical interpretation:**  
OBI of +0.60 means 80% of the inside volume is on the bid side.
OBI of −0.80 means 90% of the inside volume is on the ask side.

**Why just level 1?** The best bid/ask directly determines the next execution price.
Far-from-money orders matter less for short-horizon predictions — see
`feat_yes_obi_multilevel` for the multi-level version that uses exponential decay
across the full book depth.
