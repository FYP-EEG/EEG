# Fine hand and finger motor imagery with scalp EEG

**Author:** Anson  
**Created on:** 19/10/2026  
**Purpose:** Interim-report literature review and feasibility assessment for direct, detailed hand-movement control rather than conventional left-versus-right hand motor imagery.

## 1. Research question

Can non-invasive scalp EEG distinguish detailed motor imagery (MI) involving movements within the same hand—such as different fingers, hand opening/closing, grasping, flexion and extension—rather than only distinguishing left-hand from right-hand imagery?

## 2. Short answer

Yes, detailed same-hand MI is scientifically possible, but it is substantially harder than left-versus-right MI.

The strongest direct evidence is a 2025 *Nature Communications* study that achieved real-time robotic finger control from individual-finger MI. Across 21 experienced BCI participants, reported online accuracy was 80.56% for two fingers (thumb versus pinky) and 60.61% for three fingers (thumb, index and pinky). However, the study used a 128-channel EEG system, subject-specific deep-learning models, previous BCI experience, multiple sessions and online fine-tuning [1].

Other studies support three-finger and four/five-finger MI decoding above chance, but accuracy falls as the number of similar classes increases. Results also vary considerably across subjects and sessions [2]–[6]. Therefore, direct three-button selection is a defensible experimental aim, but it must be presented as a subject-specific feasibility study rather than a guaranteed feature.

## 3. Important distinction

The literature supports decoding **different imagined motor actions**. It does not show that EEG understands the semantic identity of an arbitrary graphical button.

A model has a plausible neural distinction when each button is associated with a distinct motor program, for example:

- Rock: imagine closing the right hand into a fist.
- Paper: imagine opening and extending the right hand.
- Scissors: imagine extending the index and middle fingers.

or:

- Rock: imagine pressing with the right thumb.
- Paper: imagine pressing with the right index finger.
- Scissors: imagine pressing with the right pinky.

The second mapping aligns especially closely with the successful thumb/index/pinky paradigm in Ding et al. [1]. By contrast, imagining the same right-index-finger click for all three buttons and changing only the button name or screen location has much weaker support in the literature.

## 4. Direct evidence from motor-imagery studies

| Study | Imagined tasks | Participants / EEG | Reported result | Relevance and limitation |
|---|---|---|---|---|
| Ding et al. (2025) [1] | Individual right-hand finger flexion/extension; online thumb versus pinky and thumb/index/pinky | 21 experienced BCI users; 128-channel BioSemi at 1024 Hz; subject-specific EEGNet and online fine-tuning | 80.56% online for two-finger MI; 60.61% for three-finger MI | Strongest proof that real-time non-invasive finger-level MI is possible. Much more data and spatial resolution than an 8-channel Cyton setup. |
| Ramu and Lakshminarayanan (2023) [2] | Left-hand thumb, index and middle finger button-pushing MI | 10 MI-naive participants; 20 electrodes at 250 Hz; features from Fz, F4, Cz, C4, Pz and P4 | 62.68 ± 6.58% without vibration; 66.31 ± 3.79% with a 150 ms fingertip vibration | Very close to the proposed “imagine pressing a button with a finger” concept. Shows three same-hand digits can be separated, although vibrotactile cueing improved performance. |
| Stankevich et al. (2016) [3] | Rhythmic MI of right thumb, index, middle and little fingers | 8 participants; key features from F3, C3 and Cz | 50 ± 7% four-class accuracy; theoretical chance 25% | Important evidence that few central channels can contain above-chance finger-MI information. Small study and moderate accuracy. |
| Alsuradi et al. (2024) [4] | Five individual finger MI classes | Public multi-subject finger-MI EEG dataset; single-trial time-domain classification | Approximately 60% within-session, 50% across sessions within subject, and 40% leave-one-subject-out | Demonstrates five-finger information but also strong session and subject variability. Supports personal models and session-specific adaptation. |
| Degirmenci et al. (2024) [5] | Five fingers plus no-mental-task state | 8 subjects; 19-channel EEG | Best six-class accuracy approximately 55% | Particularly relevant because it includes an idle state. Shows that practical false-positive control makes the problem harder. |
| Achanccaray and Hayashibe (2020) [6] | Same-hand flexion, extension and grasp MI | 20 participants; 16-channel EEG; EEGNet | 78.46 ± 12.50% for two classes and 76.72 ± 11.67% for three classes | Supports using distinct hand actions rather than arbitrary button semantics. Large between-subject variability remains. |
| Chu et al. (2020) [7] | Six unilateral upper-limb MI classes: elbow flexion/extension, wrist supination/pronation, hand close/open | 12 participants; Riemannian tangent-space features with partial least-squares reduction | 80.50% with LDA and 79.70% with SVM | Strong support for multiclass same-limb MI and for the project’s Riemannian modelling direction. Tasks include different joints/actions and are more separable than adjacent fingers. |
| Yong and Menon (2015) [8] | Rest, imaginary grasp and imaginary elbow movement | 12 participants | 66.9% for grasp versus goal-oriented elbow MI; 60.7% for three classes including rest | Shows same-limb MI is possible and that goal-oriented imagery can be more distinguishable than abstract movement imagery. |
| Alazrai et al. (2019) [9] | Eleven MI tasks involving fine parts and actions of the same hand | 18 able-bodied and 4 transradial-amputee participants; time-frequency images and CNN | 73.7% able-bodied and 72.8% amputee average accuracy | Supports rich same-hand task sets, but uses a specialised deep model and offline evaluation. |

## 5. Supporting evidence from executed movement studies

These studies demonstrate that scalp EEG contains finger-specific information, but they used actual movement rather than pure MI and therefore must not be cited as direct proof of an MI controller.

- Liao et al. decoded every pair of executed fingers from one hand with a mean accuracy of 77.11% using movement-related broadband spectral features [10].
- Lee et al. used 256 ultra-high-density electrodes over contralateral sensorimotor cortex and obtained 64.8 ± 6.3% average pairwise finger-extension accuracy [11]. The authors explicitly identify low spatial resolution as a central obstacle for conventional EEG.
- Earlier five-class work reported about 45.2% accuracy for individual executed fingers, compared with a 20% theoretical chance level, and found that broadband spectral structures were more useful than alpha or beta power alone [12].

These studies strengthen the physiological basis, but actual movement introduces movement-related potentials, proprioception and possible electromyographic contamination that are absent from pure MI.

## 6. What the results mean for this project

### 6.1 Possibility

The proposed direct three-button experiment is plausible if each class uses a distinct imagined hand or finger action. Three-class same-hand MI has been demonstrated online and offline [1], [2], [6].

### 6.2 What is not yet verified

The current “reach to and press Rock/Paper/Scissors with the same right hand” protocol has not yet been validated on the project’s participant, headband, eight-channel montage or Cyton board. Published success with 64–128 channels, dense sensorimotor coverage, large data sets and deep learning cannot be assumed to transfer to eight channels.

### 6.3 Hardware implications

The current montage—FC3, FC4, C3, Cz, C4, CP3, CP4 and Pz—covers major sensorimotor areas and includes C3/FC3/CP3 over the hemisphere contralateral to right-hand imagery. This is useful for a feasibility pilot. However, individual fingers have adjacent and overlapping cortical representations, so spatial resolution is a greater limitation than sampling rate. A 250 Hz sampling rate is sufficient for the mu/beta and 4–40 Hz information used in several relevant studies, but eight broadly spaced channels cannot reproduce a 128- or 256-channel result.

### 6.4 Model implications

- Riemannian tangent-space methods are justified by same-upper-limb evidence [7].
- Conventional binary CSP in this project cannot directly train three button classes.
- The direct-button trainer’s Riemannian logistic-regression and SVM models are reasonable first baselines.
- Deep learning should be considered only after substantially more personal data is collected. Small calibration sets are unlikely to support a reliable EEGNet model.
- Ding et al. found broadband 4–40 Hz input more informative than individual frequency bands [1]. The project should compare its current 8–30 Hz processing against a 4–40 Hz experimental variant instead of assuming 8–30 Hz is optimal for finger-level MI.

## 7. Recommended Rock–Paper–Scissors paradigm

### Option A: hand-gesture imagery

- **Rock:** kinesthetically imagine closing the right hand into a firm fist.
- **Paper:** imagine opening and fully extending the right hand.
- **Scissors:** imagine extending the index and middle fingers while the other fingers remain flexed.

This preserves an intuitive relationship between the action and the game button. It is supported indirectly by studies of grasping, hand open/close, flexion and extension [6]–[9].

### Option B: individual-finger pressing imagery

- **Rock:** imagine pressing the Rock button repeatedly with the right thumb.
- **Paper:** imagine pressing the Paper button repeatedly with the right index finger.
- **Scissors:** imagine pressing the Scissors button repeatedly with the right pinky.

This is more directly aligned with the thumb/index/pinky study that achieved 60.61% online three-class accuracy [1]. Thumb and pinky are generally preferable to adjacent fingers because adjacent-finger cortical patterns overlap strongly.

### Recommendation

Test Option B first as the scientific feasibility experiment because it most closely reproduces the strongest published paradigm. Option A can then be compared as a more intuitive user-interface mapping.

## 8. Proposed validation protocol

### Phase 1: prove a binary distinction

1. Use right-thumb versus right-pinky press imagery.
2. Record at least 40 trials per class in each of three separate sessions.
3. Use kinesthetic imagery: feel the finger flex and the imagined contact/pressure, without moving.
4. Use a 3–4 s imagery period and at least 2 s rest.
5. Monitor the real hand with forearm/finger EMG if possible. At minimum, record video and reject visible movement trials.
6. Evaluate with session-held-out cross-validation, not random window splitting.

### Phase 2: add the third class

1. Add right-index-finger pressing.
2. Record balanced thumb/index/pinky trials over at least three sessions.
3. Report balanced accuracy, per-class precision/recall, confusion matrix and a permutation-test threshold.
4. Evaluate an idle period separately and report false activations per minute.

### Suggested project decision gates

These are engineering targets, not universal scientific thresholds:

- **Binary pilot:** proceed if session-held-out accuracy is statistically above chance and approximately 65% or better.
- **Three-class pilot:** proceed if session-held-out balanced accuracy is statistically above the 33.3% chance level and approximately 50% or better.
- Reject the approach for the current hardware/user if performance remains near permutation chance after three clean sessions.

A high within-session score alone is insufficient. Five-finger research shows a large drop from within-session to cross-session and cross-subject performance [4].

## 9. Report-ready conclusion

> Existing research demonstrates that scalp EEG motor imagery is not limited to distinguishing left- and right-hand intentions. Studies have decoded different movements within the same limb, including grasping, flexion and extension, hand opening and closing, and individual finger movements. Most notably, Ding et al. demonstrated real-time non-invasive robotic finger control with mean accuracies of 80.56% for two-finger motor imagery and 60.61% for three-finger motor imagery. However, fine-movement decoding is more difficult than conventional left-versus-right motor imagery because individual fingers have adjacent and overlapping cortical representations. Performance depends strongly on the participant, recording session, number and placement of electrodes, quantity of calibration data and use of subject-specific adaptation. Therefore, direct three-button control is scientifically plausible when each button is mapped to a distinct imagined finger or hand action, but its feasibility with an eight-channel Cyton system must be established experimentally using multi-session, leakage-safe validation.

## 10. References

[1] Y. Ding, C. Udompanyawit, Y. Zhang, and B. He, “EEG-based brain-computer interface enables real-time robotic hand control at individual finger level,” *Nature Communications*, vol. 16, art. 5401, 2025. https://doi.org/10.1038/s41467-025-61064-x

[2] V. R. V. Ramu and K. L. K. Lakshminarayanan, “Enhanced motor imagery of digits within the same hand via vibrotactile stimulation,” *Frontiers in Neuroscience*, vol. 17, art. 1152563, 2023. https://doi.org/10.3389/fnins.2023.1152563

[3] L. A. Stankevich et al., “EEG pattern decoding of rhythmic individual finger imaginary movements of one hand,” *Human Physiology*, vol. 42, pp. 32–42, 2016. https://doi.org/10.1134/S0362119716010175

[4] H. Alsuradi, A. Khattak, A. Fakhry, and M. Eid, “Individual-finger motor imagery classification: a data-driven approach with Shapley-informed augmentation,” *Journal of Neural Engineering*, vol. 21, art. 026013, 2024. https://doi.org/10.1088/1741-2552/ad33b3

[5] M. Degirmenci, Y. K. Yuce, M. Perc, and Y. Isler, “EEG-based finger movement classification with intrinsic time-scale decomposition,” *Frontiers in Human Neuroscience*, vol. 18, art. 1362135, 2024. https://doi.org/10.3389/fnhum.2024.1362135

[6] D. Achanccaray and M. Hayashibe, “Decoding hand motor imagery tasks within the same limb from EEG signals using deep learning,” *IEEE Transactions on Medical Robotics and Bionics*, vol. 2, no. 4, pp. 692–699, 2020. https://doi.org/10.1109/TMRB.2020.3025364

[7] Y. Chu et al., “Decoding multiclass motor imagery EEG from the same upper limb by combining Riemannian geometry features and partial least squares regression,” *Journal of Neural Engineering*, vol. 17, art. 046029, 2020. https://doi.org/10.1088/1741-2552/aba7cd

[8] X. Yong and C. Menon, “EEG classification of different imaginary movements within the same limb,” *PLOS ONE*, vol. 10, no. 4, e0121896, 2015. https://doi.org/10.1371/journal.pone.0121896

[9] R. Alazrai, M. Abuhijleh, H. Alwanni, and M. I. Daoud, “A deep learning framework for decoding motor imagery tasks of the same hand using EEG signals,” *IEEE Access*, vol. 7, pp. 109612–109627, 2019. https://doi.org/10.1109/ACCESS.2019.2934018

[10] K. Liao et al., “Decoding individual finger movements from one hand using human EEG signals,” *PLOS ONE*, vol. 9, no. 1, e85192, 2014. https://doi.org/10.1371/journal.pone.0085192

[11] H. S. Lee et al., “Individual finger movement decoding using a novel ultra-high-density electroencephalography-based brain-computer interface system,” *Frontiers in Neuroscience*, vol. 16, art. 1009878, 2022. https://doi.org/10.3389/fnins.2022.1009878

[12] K. Liao et al., “Evaluation of EEG features in decoding individual finger movements from one hand,” *Computational Intelligence and Neuroscience*, vol. 2013, art. 243257, 2013. https://doi.org/10.1155/2013/243257
