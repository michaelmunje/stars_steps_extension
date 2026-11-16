## Headline

* **ETH/UCY leave-one-out, minADE20/minFDE20 (5-fold avg, m).** Leak-free rows: MoFlow 0.22/0.34, AgentFormer 0.23/0.39, SocialVAE 0.25/0.44, **STARS LOO 0.25/0.47**, **STARS pooled w/o ETH/UCY (zero-shot) 0.25/0.47**, STARS STEPS-only (zero-shot) 0.27/0.49, Trajectron++ (leak-free features) 0.34/0.61, Social-GAN 0.36/0.64, CV 0.53/1.15. STARS with an oracle goal reaches 0.22/0.34 (privileged input). STARS is competitive with SocialVAE on best-of-20 but behind MoFlow/AgentFormer, mostly on FDE.
* **Single-sample accuracy.** The mean-over-samples ADE of STARS CVAE (0.56) is the best of the learned models (SocialVAE 0.66, T++ 0.73, SGAN 0.86, MoFlow 0.99, AgentFormer > 1) and close to CV (0.53): the best-of-20 leaders buy their minADE with widely spread samples.
* **Collisions between predicted agents.** On ETH/UCY, 8.1 % of real agents come within 0.4 m of another (0.5 % within 0.2 m). STARS CVAE predictions do so for 14 % (5 % within 0.2 m) -- the fewest of all methods (CV 18 %, MoFlow 20 %, AgentFormer 20 %, SocialVAE 23 %, T++ 24 %, SGAN 24 %), but still well above the data, especially at 0.2 m. STARS' diffusion decoders are the worst (28-30 %). On our datasets the ordering is the same; STARS matches the GT contact rate on STEPS (5.4 % vs 5.8 %) and over-predicts contacts most on grand_central / ATC.
* **Our datasets.** STARS pooled (w/ or w/o ETH/UCY) is best or tied-best of the non-oracle models on every dataset except STEPS (all methods ~0.07 minADE) and ucy (STARS pooled 0.19/0.39 vs MoFlow-STEPS 0.20/0.37).
* **Closed loop (up to 20 s).** Receding-horizon STARS (no-ETH/UCY, 1.6 s steps) has the lowest ADE among learned models at every horizon (2.63 m at 20 s; SocialVAE 2.73, T++ 3.95, AgentFormer 3.89, SGAN 4.16; CV 2.54) and the lowest collision rate, close to GT (0.27 vs 0.33 at 20 s). STARS pooled native (1-step loop) is best overall at 20 s (2.39 m). The LOO fine-tuned STARS drifts in its native 1-step loop (speeds rise to 0.74 m/s vs 0.24 GT): fine-tuning used only the open-loop loss. MoFlow's closed loop is still running (pending row).
* **STARS improved (branch `stars-improve`, `results/stars_improve_2026-11-15`).** Two new rows.
  * **`stars_improved`:** the pooled recipe on all 7 datasets, plus inference-time selection of joint samples. It draws 4x20 samples and keeps the 20 with the fewest close pairs, ignoring pairs that are walking groups in the observation.
    * It keeps `stars_pooled`'s accuracy on ETH/UCY (minADE20 0.23/0.43, mean-over-samples 0.55/1.15) and on our datasets.
    * Predicted contacts on ETH/UCY fall from 5.0% to 3.1% within 0.2 m and from 14.2% to 10.1% within 0.4 m (GT 0.5% / 8.1%).
    * On our datasets, contacts within 0.2 m drop by 15-70%: least on grand_central, most on eth and ucy.
  * **`stars_improved_social`:** adds a decoder refinement round that sees the neighbours' predicted futures, plus a collision hinge loss.
    * It reaches GT contact rates on ETH/UCY: 0.6% within 0.2 m and 4.5% within 0.4 m (GT 0.5% / 8.1%).
    * The cost is accuracy: minADE20 0.27/0.51 and a slightly slower speed.
  * Both share `stars_pooled`'s ETH/UCY leakage: hotel, zara1 and univ were in training.

