# Sensorimotor nodes, actions and EEG channel relevance

**Author:** Anson  
**Created on:** 19/10/2026  
**Purpose:** Academic-resource review of the cortical nodes involved in motor imagery, the actions/functions associated with them, and their approximate relationship to the project’s eight EEG electrodes.

## 1. Essential clarification: a cortical node is not an EEG electrode

A **cortical node** is a brain region participating in a functional network—for example, primary motor cortex (M1), supplementary motor area (SMA), premotor cortex (PMC), primary somatosensory cortex (S1), or posterior parietal cortex (PPC).

An **EEG electrode** is a scalp sensor. It records a spatially mixed projection from multiple cortical sources because electrical activity spreads through brain tissue, cerebrospinal fluid, skull and scalp. Therefore:

- C3 is not a direct measurement of one “right-hand neuron group”.
- FC3 is not exclusively premotor cortex.
- CP3 is not exclusively somatosensory cortex.
- No scalp electrode corresponds uniquely to Rock, Paper, Scissors, the thumb, or another finger.

The international 10–10 system provides approximate anatomical coverage, not one-to-one cortical localization. MRI-based studies report several millimetres of variability in cortical projections, with especially high variability in central and parietal positions [1], [2]. A separate EEG–fMRI study found conventional electrodes such as C3/C4 could lie 16–42 mm from the corresponding fMRI-defined functional area [3].

This means the correct scientific language is:

> “The classifier uses a multichannel spatial pattern sensitive to a sensorimotor network.”

It should not be described as:

> “C3 is the thumb node and CP3 is the index-finger node.”

## 2. The motor-imagery network

Meta-analyses show that MI is supported by a distributed frontoparietal network rather than M1 alone. Consistently involved regions include bilateral premotor cortex, SMA/pre-SMA, superior and inferior parietal regions, basal ganglia and cerebellum; M1 involvement is weaker and less consistent than during executed movement [4], [5].

### Main cortical and subcortical nodes

| Node | Main purpose in action/MI | Actions for which it is especially relevant | Approximate scalp sensitivity |
|---|---|---|---|
| **Primary motor cortex (M1; BA4), including the hand knob** | Final cortical motor output; movement direction, force and muscle-synergy representation. Activity during MI is weaker and less consistent than during execution. | Finger flexion/extension, hand opening/closing, grasp force, executed button press; imagined versions may modulate the same network more weakly. | Central electrodes around C1/C3 for right-hand actions and C2/C4 for left-hand actions. Exact correspondence varies by person. |
| **Primary somatosensory cortex (S1; BA3/1/2)** | Represents tactile and proprioceptive state of the hand; contributes to kinesthetic feeling, imagined contact and movement feedback. | Imagined button contact/pressure, finger position, hand shape and tactile imagery. | Central/postcentral and centroparietal sensors: C3/CP3 for the right hand; C4/CP4 for the left hand. |
| **Dorsal premotor cortex (PMd; BA6)** | Selects and prepares movements using target, direction and spatial information; important for visually guided reach planning. | Reaching toward left/centre/right targets, directional arm or hand movement, preparing a finger action from a cue. | Fronto-central sensors such as FC1/FC3/FC5 on the left and FC2/FC4/FC6 on the right. |
| **Ventral premotor cortex (PMv; BA6)** | Selects hand posture and grip configuration; participates in object-related grasp planning and action observation. | Precision grip, power grip, hand preshaping, fist/open-hand/scissors gestures. | More lateral fronto-central positions such as FC5/FC6; conventional eight-channel montages may cover this incompletely. |
| **Supplementary motor area and pre-SMA (medial BA6)** | Internally generated action, movement initiation, sequence organisation, rhythm and bimanual coordination; MI often relies strongly on SMA. | Repetitive finger MI, ordered finger tapping, multi-step gestures, self-paced imagery. | Midline fronto-central region, ideally FCz and Cz. The current montage has Cz but not FCz. |
| **Anterior intraparietal sulcus (aIPS) / inferior parietal lobule (IPL)** | Transforms object size/shape into hand preshaping and grip configuration; codes grasp-related goals. | Grasp type, finger aperture, imagined object manipulation and hand-shape selection. | Lateral centroparietal/parietal positions such as CP3/CP5/P3 on the left and CP4/CP6/P4 on the right; localization is approximate. |
| **Superior parietal lobule / medial intraparietal areas** | Converts visual, gaze, proprioceptive and body coordinates into reaching trajectories; monitors online position. | Reaching to spatially separated monitor buttons, pointing and trajectory imagery. | CP1/CP3/P1/P3/Pz for right-hand actions and homologous right-hemisphere positions for left-hand actions. |
| **Dorsolateral prefrontal cortex (DLPFC)** | Maintains the instruction/goal, attention, rule and working memory; controls the task rather than encoding the movement itself. | Remembering which button or gesture is required, suppressing actual movement, task switching. | Frontal electrodes such as F3/F4. The current montage has no dedicated frontal cognitive channels. |
| **Basal ganglia** | Action selection, initiation and learned sequencing. | Choosing and initiating a trained motor program. | Deep source; not directly localized by an eight-channel scalp montage. |
| **Cerebellum** | Forward prediction, timing, error correction and internal simulation. | Rhythmic/repetitive finger imagery, learned trajectories, timing. | Deep/posterior source; not directly assignable to one conventional scalp electrode. |

## 3. Action-specific network organisation

### 3.1 Individual fingers

Individual fingers are not stored in five clean, isolated M1 patches. High-resolution fMRI shows an ordered somatotopic gradient, but the representations are distributed and substantially overlapping [6], [7]. Direct mapping of single digits against rest produced large overlaps; clearer separation appeared only when one finger was contrasted directly with another. Reported inter-digit distances can be only a few millimetres, especially for adjacent fingers [6].

Consequences for EEG:

- Thumb versus pinky is usually more plausible than index versus middle.
- Adjacent fingers are likely to be confused.
- Finger identity must be decoded from a multichannel pattern, not from one electrode.
- Eight-channel scalp EEG cannot spatially resolve millimetre-scale finger maps directly; classification must exploit temporal, spectral and covariance differences distributed across the network.

The M1 hand region itself is associated with the characteristic precentral “hand knob” [8]. However, an electrode position is not guaranteed to sit exactly over that landmark in every participant.

### 3.2 Hand opening, closing and grasping

Hand shape is represented across a parieto-premotor-motor network:

1. aIPS/IPL represents object geometry and desired hand preshape;
2. PMv and PMd select/configure the grip and transport action;
3. M1 contributes execution-related motor output;
4. S1 contributes imagined or actual hand state and contact.

Human studies describe partly overlapping but functionally differentiated parietofrontal circuits for reaching, grasping and pointing [9], [10]. For MI specifically, aiming, squeezing and extension–flexion actions were decodable from patterns in PMC, M1, SPL, IPL and IPS [11].

Therefore, Rock/fist, Paper/open-hand and Scissors/two-finger-extension are more plausibly different because they recruit different hand-shape and motor-program patterns—not because each has one unique scalp node.

### 3.3 Reaching toward a monitor button

Reaching is especially dependent on PMd and superior/medial PPC because the brain must transform target position, eye position and current hand/body state into a trajectory [9], [10]. Grasping or pressing additionally engages anterior/lateral parietal and ventral premotor circuitry.

For a right-hand imagined reach:

- **FC3 region:** motor preparation and target/direction-related premotor contribution;
- **CP3/Pz region:** spatial and sensorimotor transformation contribution;
- **C3 region:** hand/finger motor and sensorimotor rhythm contribution.

However, using three visual target locations also introduces visual attention and eye-movement differences. Those must be controlled so the classifier is not merely learning gaze direction or cue position.

### 3.4 Sequential or rhythmic finger actions

SMA/pre-SMA and premotor networks are important for internal generation and sequencing. Effective-connectivity research on imagined finger tapping found stronger PMC→SMA and M1→PMC coupling during MI than execution, illustrating that imagery depends on network interactions rather than one isolated node [12].

### 3.5 Imagined touch and button pressure

Kinesthetic MI includes the felt movement and expected sensory consequence. S1 and posterior sensorimotor regions are therefore relevant to imagined button contact, pressure and finger position. Mu/beta ERD is a frequency-specific network response rather than a direct readout of muscle commands [13]. Tactile-imagery studies also report contralateral sensorimotor ERD, supporting the use of imagined contact as part of the task [14].

## 4. Interpreting the project’s current eight-channel montage

Current order:

```text
FC3, FC4, C3, Cz, C4, CP3, CP4, Pz
```

### Approximate functional interpretation

| Channel | Approximate network sensitivity | Most relevant project use | Important warning |
|---|---|---|---|
| **FC3** | Left frontal/premotor region, often near BA6 | Right-hand/finger planning; imagined reach and action selection | Not a forehead/EOG channel and not a pure blink channel. It contains useful motor information. |
| **FC4** | Right frontal/premotor region | Left-hand planning; bilateral/control information during right-hand tasks | Also not a pure artifact channel. |
| **C3** | Left central sensorimotor region; may project to precentral or postcentral cortex | Main right-hand/finger sensorimotor ROI; mu/beta ERD | “C3 = M1 hand” is an approximation. One MRI study found C3 most often projected to postcentral rather than precentral cortex [2]. |
| **Cz** | Midline central/SMA-adjacent and bilateral sensorimotor contribution | Movement onset, sequence, bilateral/common activity | Not exclusively feet; it can contribute to hand sequencing and general movement preparation. |
| **C4** | Right central sensorimotor region | Main left-hand ROI; ipsilateral/control information for right-hand imagery | Same localization caveat as C3. |
| **CP3** | Left posterior sensorimotor/parietal region | Right-hand kinesthesia, imagined contact, hand state and reach/grasp transformation | May reflect several parietal regions because of volume conduction and anatomical variability. |
| **CP4** | Right posterior sensorimotor/parietal region | Left-hand kinesthesia; control/bilateral pattern | Not a one-to-one S1 measurement. |
| **Pz** | Midline posterior parietal/precuneus-biased region | Spatial attention, target/reach context and network-level information | Less directly motor than FC/C channels and vulnerable to visual-task confounds. |

MRI-based electrode localisation demonstrates why these descriptions must remain probabilistic. In one fixed-cap study, C1 projected mainly to precentral gyrus/BA4, whereas C3 projected mainly to postcentral gyrus/BA1; central and parietal positions varied substantially across participants [2].

## 5. Recommended channel groups for analysis

For the existing montage, use channel groups as **regions of interest**, not individual action labels:

```text
Right-hand contralateral ROI: FC3, C3, CP3
Left-hand contralateral ROI:  FC4, C4, CP4
Midline/network ROI:          Cz, Pz
```

For right-hand-only fine MI:

- the left ROI should contain the strongest motor-relevant information;
- the right ROI should be retained because MI can be bilateral and because it provides useful covariance/control information;
- Cz and Pz can add sequence, common motor and parietal context;
- classification should use all-channel patterns, covariance or connectivity rather than “highest power at one node”.

## 6. Mapping proposed commands to networks

| Proposed command | Intended mental action | Expected dominant network—not a unique electrode | Current channels most likely to contribute |
|---|---|---|---|
| **Rock** | Kinesthetically close right hand into a fist / power grasp | aIPS/IPL + PMv/PMd + left M1/S1 + SMA | FC3, C3, CP3, Cz; possibly Pz |
| **Paper** | Open and extend right hand/fingers | Left premotor + M1/S1; parietal hand-state representation | FC3, C3, CP3, Cz |
| **Scissors** | Extend index and middle fingers while others remain flexed | Fine finger pattern in left M1/S1 plus premotor sequencing/shape network | FC3, C3, CP3, Cz; distinctions may be subtle |
| **Spatial button press** | Reach to a target and press with a specified finger | PMd/SPL for reach + aIPS/PMv for press/grip + M1/S1 for finger | FC3, C3, CP3, Pz, Cz |
| **Thumb/index/pinky direct selection** | Repetitive imagined flexion/press of one specified finger | Distributed left hand-knob and somatosensory pattern | FC3, C3, CP3, Cz; no electrode uniquely owns a finger |

## 7. Implications for montage design

The current symmetric montage is sensible for conventional left-versus-right MI. For right-hand-only finger or gesture decoding, its limitation is that only FC3, C3 and CP3 densely sample the contralateral hemisphere.

If electrode placement can be changed for a dedicated right-hand fine-MI experiment, literature supports testing denser left sensorimotor coverage around positions such as:

```text
FC5, FC3, FC1, C5, C3, C1, CP5, CP3/CP1
```

This is a hypothesis-driven experimental montage, not a guaranteed optimal layout. Individual anatomy and mu-rhythm topography vary, and central/parietal electrode projections have measurable between-person variability [1], [2]. A separate EOG electrode should be used for blink monitoring rather than sacrificing FC3/FC4 motor channels.

## 8. Recommended feature interpretation

### Mu and beta ERD/ERS

Movement and MI commonly modulate sensorimotor mu (~8–13 Hz) and beta (~13–30 Hz) rhythms. ERD is a reduction in band-limited synchrony relative to baseline; ERS is an increase. These effects are frequency-specific and may coexist at different scalp locations [13].

### Movement-related cortical potentials

Slow cortical potentials/MRCPs can reflect preparation and movement onset. Fronto-central, central and centroparietal electrodes are commonly used, but slow signals are also susceptible to drift and eye/movement artifacts.

### Covariance and connectivity

Fine actions share the same gross sensorimotor region, so absolute power at C3 alone may be similar across classes. More useful information may lie in:

- covariance among FC3–C3–CP3–Cz;
- time-varying network coupling;
- broadband temporal/spectral structure;
- source-space or spatially filtered patterns.

This supports retaining Riemannian tangent-space models for the first direct-button experiments.

## 9. Report-ready conclusion

> Motor imagery should be modelled as activity within a distributed sensorimotor network rather than as a one-to-one mapping between actions and EEG electrodes. Primary motor and somatosensory cortices contribute movement- and body-state information, premotor regions contribute action preparation and hand configuration, supplementary motor regions contribute internally generated sequencing, and posterior parietal regions transform visual, proprioceptive and goal information into reach and grasp plans. Although the 10–10 electrode positions FC3, C3 and CP3 provide useful contralateral coverage for right-hand imagery, each scalp sensor records a spatial mixture of several cortical sources and varies anatomically between participants. Individual finger representations are ordered but strongly overlapping at the cortical level, so detailed hand actions must be decoded from multichannel spatiotemporal patterns rather than assigned to individual “finger electrodes.”

## 10. References

[1] L. Koessler et al., “Automated cortical projection of EEG sensors: Anatomical correlation via the international 10–10 system,” *NeuroImage*, vol. 46, no. 1, pp. 64–72, 2009. https://doi.org/10.1016/j.neuroimage.2009.02.006

[2] C. L. Scrivener and A. T. Reader, “Variability of EEG electrode positions and their underlying brain regions: visualizing gel artifacts from a simultaneous EEG-fMRI dataset,” *Brain and Behavior*, vol. 12, no. 2, e2476, 2022. https://doi.org/10.1002/brb3.2476

[3] V. L. Ives-Deliperi and J. T. Butler, “Relationship between EEG electrode and functional cortex in the international 10 to 20 system,” *Journal of Clinical Neurophysiology*, vol. 35, no. 6, pp. 504–509, 2018. https://doi.org/10.1097/WNP.0000000000000510

[4] S. Hétu et al., “The neural network of motor imagery: An ALE meta-analysis,” *Neuroscience & Biobehavioral Reviews*, vol. 37, no. 5, pp. 930–949, 2013. https://doi.org/10.1016/j.neubiorev.2013.03.017

[5] R. M. Hardwick, S. Caspers, S. B. Eickhoff, and S. P. Swinnen, “Neural correlates of action: Comparing meta-analyses of imagery, observation, and execution,” *Neuroscience & Biobehavioral Reviews*, vol. 94, pp. 31–44, 2018. https://doi.org/10.1016/j.neubiorev.2018.08.003

[6] P. Dechent and J. Frahm, “Functional somatotopy of finger representations in human primary motor cortex,” *Human Brain Mapping*, vol. 18, no. 4, pp. 272–283, 2003. https://doi.org/10.1002/hbm.10084

[7] P. Hluštík et al., “Somatotopy in human primary motor and somatosensory hand representations revisited,” *Cerebral Cortex*, vol. 11, no. 4, pp. 312–321, 2001. https://doi.org/10.1093/cercor/11.4.312

[8] T. A. Yousry et al., “Localization of the motor hand area to a knob on the precentral gyrus: A new landmark,” *Brain*, vol. 120, pp. 141–157, 1997. https://doi.org/10.1093/brain/120.1.141

[9] F. Filimon, “Human cortical control of hand movements: Parietofrontal networks for reaching, grasping, and pointing,” *The Neuroscientist*, vol. 16, no. 4, pp. 388–407, 2010. https://doi.org/10.1177/1073858410375468

[10] G. Vingerhoets, “Contribution of the posterior parietal cortex in reaching, grasping, and using objects and tools,” *Frontiers in Psychology*, vol. 5, art. 151, 2014. https://doi.org/10.3389/fpsyg.2014.00151

[11] S. Pilgramm et al., “Motor imagery of hand actions: Decoding the content of motor imagery from brain activity in frontal and parietal motor areas,” *Human Brain Mapping*, vol. 37, no. 1, pp. 81–93, 2016. https://doi.org/10.1002/hbm.23015

[12] Y. K. Kim, E. Park, A. Lee, C.-H. Im, and Y.-H. Kim, “Changes in network connectivity during motor imagery and execution,” *PLOS ONE*, vol. 13, no. 1, e0190715, 2018. https://doi.org/10.1371/journal.pone.0190715

[13] G. Pfurtscheller and F. H. Lopes da Silva, “Event-related EEG/MEG synchronization and desynchronization: Basic principles,” *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, 1999. https://doi.org/10.1016/S1388-2457(99)00141-8

[14] E. S. Shkurko et al., “Event-related desynchronization induced by tactile imagery: An EEG study,” *eNeuro*, vol. 10, no. 6, 2023. https://doi.org/10.1523/ENEURO.0455-22.2023
