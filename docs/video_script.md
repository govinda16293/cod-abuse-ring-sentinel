# 5-minute pitch video — script

Spoken lines are what I say out loud. `SCREEN:` is what is showing at that beat.

Every number below is pulled from a file in `artifacts/` or `data/`. Do not round
on camera. Do not say "roughly" or "about". Say the number.

**Before you record.** Run `streamlit run app/streamlit_app.py`. If you want the
language model to write the justification live, export `ANTHROPIC_API_KEY` first.
Without a key the button still works and produces the deterministic template
version, which demonstrates the same boundary. Say on camera which one you are
showing.

**One exception, flagged.** At 4:15 the pre-bugfix R2 recall of 0.918 and the
pre-bugfix precision of 0.9153 are historical values recorded in `LOG.md` at the
time. They are not re-derivable from the current artifacts, because the bug is
fixed and the model was retrained. If asked, say they are logged, not recomputed.
Everything else is checkable against a file.

---

## 0:00 – 0:45 · The problem, in rupees

`SCREEN:` Streamlit **Order detail** tab, one flagged order open, evidence panel
visible.

- A customer takes delivery of a cash-on-delivery order.
- They file a wrong-item return and get cash back at the door.
- The goods that come back are worthless.
- The merchant paid to ship it out, paid to pick it up, and burned ops time.
- On a two thousand rupee order, that miss costs one thousand six hundred and
  forty rupees.
- Wrongly blocking a real customer costs four hundred and forty-one rupees forty.
- So the two errors are not equal. The ratio is three point seven two to one.
- F1 assumes they are equal. That is why I did not use it.
- One customer doing this is a bad customer. I did not go after those.
- I went after rings. Groups of accounts running the same play, sharing a device
  or a flat or a block of phone numbers.
- Each account looks ordinary alone. The evidence only exists between them.

## 0:45 – 1:45 · Architecture, walking the diagram

`SCREEN:` the ASCII diagram in `ARCHITECTURE.md`, then the `[leak] PASS` line in
the terminal.

- Six phases. Left to right on this diagram.
- Orders and returns go in. They become timestamped events, replayed in order.
- A return only becomes visible at its return timestamp, not at its order
  timestamp.
- That gives me a customer, device, address and phone graph as of each order.
- Those features feed LightGBM, then isotonic calibration, and out comes a
  probability.
- The probability meets a cost curve in rupees, which picks three bands.
- TreeSHAP explains each flag. The language model writes it up.
- Here is the part I care about most.
- There is no global graph pass followed by a split. A global graph already
  contains edges built by test-set orders.
- So I prove it instead of claiming it. This deletes every row after a cutoff
  date and recomputes every feature from the truncated world.
- If a feature had read a future row, deleting that row would move it.
- Sixty-three features, bit-identical, across two hundred and seventy-two
  thousand nine hundred and fifty-nine orders.
- That runs as a gate before any model is trained.
- Sixty-three features are computed. Forty-nine are used.
- I chose LightGBM over a graph neural network on purpose. The graph is mostly
  stars and small cliques. It is already summarised into about fifteen component
  features. And a tree gives me exact attribution I can defend line by line.
- The language model never sees the label and never sees the threshold. The
  classifier decides. The model writes.

## 1:45 – 3:15 · Live demo, real batch

`SCREEN:` Streamlit, **Batch** tab first, then **Review queue**, then **Order
detail**.

- This is the held-out test set. Forty-nine thousand nine hundred and five COD
  orders, scored.
- Ninety-four point nine two percent pass straight through as cash on delivery.
- Two point three two percent go to a human queue.
- Two point seven six percent get auto-actioned. That means forced to prepay.
- The cut points are zero point zero three four three and zero point three four
  four one. Both were chosen on validation, never on this test set.

`SCREEN:` switch to **Review queue**.

- This is what an analyst actually opens in the morning. Sorted by score.

`SCREEN:` switch to **Order detail**. It opens on order 239747 by default, which
is the highest-scoring flagged order. Do not pick a different one on camera; every
number below is that order's.

- Order 239747. Auto-action. Calibrated probability one point zero zero zero zero.
- Four thousand one hundred and sixty-eight rupees, footwear, cash on delivery.
- The evidence comes straight from TreeSHAP, rendered as plain sentences.
- Four point five new account-to-account links formed in that cluster in the past
  week. That is the strongest single contributor.
- The account is forty-eight days old. It sits in an identity cluster of nine
  linked accounts. The average account in that cluster is fifty-four days old.
- The cluster placed the equivalent of five point seven orders in the past week.
- The held-out label agrees, and the ring signature is R2, address fuzzing.
- Now the part I would want to see if I were the reviewer. What this does not
  establish.
- For this order it says the account has no order history, so every behavioural
  signal is absent and the score rests entirely on network structure.
- That is generated from the feature row by rule. I did not write it by hand, and
  it is the caveat that actually applies to this order.
- And this is the justification the language model writes, from that payload
  only.
- It has no path back into the decision. If I had asked a model "is this fraud",
  I would have a second uncalibrated classifier wearing a justification costume.

## 3:15 – 4:15 · Metrics

`SCREEN:` Streamlit **Metrics** tab, then the cost curve, then the
recall-by-ring-type table.

- Start with the money, because that is the point.
- No detector at all costs one hundred and fifty-five thousand four hundred and
  one rupees per thousand COD orders.
- Switching cash on delivery off entirely costs four hundred and sixty thousand
  four hundred and fifty-six. Three times worse.
- A single cost-optimal threshold costs twenty-five thousand nine hundred and
  twenty-six.
- The three-band policy I shipped costs eleven thousand six hundred and
  seventeen. That is a ninety-two point five two percent reduction.
- Underneath that: PR-AUC zero point eight eight zero nine. Precision zero point
  nine three one six. Recall zero point eight zero seven two.
- Calibration error zero point zero zero one eight, which is what lets me
  threshold on the probability at all.
- Now the obvious objection. That whole result rests on cost constants I picked.
- Five of the eleven have no published figure anywhere. So I swept all of them.
- Twenty-eight cost worlds. Both halves of each error cost at plus and minus
  twenty-five and fifty percent. Reviewer accuracy seventy to ninety-five.
  Review capacity two to ten percent.
- Each one re-runs threshold and band selection from scratch, on validation.
- The three-band structure survives in twenty-eight out of twenty-eight. It beats
  the single threshold in twenty-eight out of twenty-eight. It beats doing
  nothing in twenty-eight out of twenty-eight.
- Worst case, if I tuned to the wrong costs, I lose seven hundred and forty-four
  rupees sixty-six per thousand orders. That is zero point five three percent of
  the loss the detector avoids.
- One place it does break, and I will name it. Review capacity below two point
  one three percent makes the shipped policy infeasible, not just worse.

`SCREEN:` recall-by-ring-type table.

- And this is the table I would not want hidden behind a single recall number.
- Phone-block rings, zero point nine eight two eight. Address-fuzzing rings, zero
  point nine seven six seven. Shared-device, zero point eight eight seven five.
- Burst-value rings, zero point three seven five five. Another zero point four
  nine four one go to human review.
- Those share no identifier at all, by construction. All that is left is new
  account, cash on delivery, above-average basket. That also describes a
  legitimate first-time customer.
- That is an information limit. The evidence is not in an order log.

## 4:15 – 5:00 · What broke

`SCREEN:` `LOG.md`, scrolling, then the flat-signup comparison table in the
README.

- Two things broke, and both are in the log with dates.
- First one. A unit test on my address normaliser failed. Four spellings of one
  flat were producing two cluster keys.
- The word "apt" was in both my filler list and my synonym list. I expanded
  synonyms first. So "Apt 4B" turned into a building-name token instead of being
  stripped as a prefix.
- One physical door, two clusters, graph edge silently gone.
- Address-fuzzing recall went from zero point nine one eight to zero point nine
  seven six seven. Precision from zero point nine one five three to zero point
  nine three one six.
- I only found that because I tested the normaliser on its own. End to end, the
  pipeline ran clean and the numbers looked fine.
- Second one, and this is the bigger one.
- I ran a temporal split. Train on months one to twelve, test on thirteen to
  eighteen. Precision fell from zero point nine three one six to zero point three
  nine nine four.
- I diagnosed it. Ninety-nine point six six percent of the two thousand and
  sixty-five false positives were customers never seen in training. Median tenure
  ten point nine two days. Median cluster size one. No graph links at all.
- I wrote that up as the main finding of the project.
- Then I checked my own generator. Order count per customer never scaled with how
  long the account had existed.
- So someone who signed up five days before the window closed still drew a full
  year of orders and placed all of them in those five days.
- That manufactured exactly the population the model over-flags.
- I regenerated once with that fixed. Precision came back to zero point nine one
  nine seven.
- Most of the collapse I reported was my own bug.
- What survives is zero point zero two one nine of PR-AUC over six months, and
  that is an upper bound.
- I left the original number standing next to the correction, because the
  sequence is the point.

---

## Delivery notes

- Do not round. Do not say "roughly" or "about".
- The strongest ninety seconds are 3:15 to 5:00. If you are running long, cut the
  architecture beat, not the metrics or the correction.
- If asked why not a GNN, the answer is the docstring at the top of
  `src/train.py`. Do not improvise it.
- If asked whether the test set was touched: it was scored three times, all
  before the final freeze, and no threshold, band, feature set or hyperparameter
  was ever selected on it.
