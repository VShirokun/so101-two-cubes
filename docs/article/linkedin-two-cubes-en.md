# Two cubes with markers: how a real robot arm calibrated itself, recorded its own dataset, and lifted a cube with a neural network

*LinkedIn draft. Media files are in `docs/article/media/`; numbers in brackets refer to them.*

We have an SO-101 manipulator (shoulder, elbow, wrist, gripper, two webcams: one above the table, one on the wrist) and a goal: teach it to pick things up from images, the way VLA policies like GR00T do. Every project like this hits the same wall: data. We have no teleoperator, and without hundreds of demonstrations there is nothing to learn from. On top of that, a real arm does not know where its cameras are, and without that no automatic data collector will ever land the fingers on a cube.

The fix turned out to be surprisingly cheap: two 3D-printed cubes with ArUco markers on all six faces. They are both the object the robot learns to grasp and the measuring instrument it uses to calibrate itself. Here is the whole path, two weeks of it, with numbers and videos.

## 1. Cubes you cannot assemble wrong

[01, 02] The cube is 28 mm, an exact twin of the cube in our MuJoCo simulator. Each face carries its own marker from the DICT_4X4_50 dictionary: IDs 1–6 on the first cube, 7, 10, 11, 13, 14, 15 on the second. Markers on all six faces mean the cube is readable in any pose: lying on the table, squeezed between the fingers, or held up in front of the wrist camera.

Two-colour printing: a white body with shaped pockets 0.8 mm deep and six black "key" inlays. Each pocket's outline is unique to its ID and has no rotational symmetry, so an inlay physically cannot be inserted the wrong way. Only IDs whose black cells are all edge-connected were used; otherwise an inlay would fall apart into pieces. The black PLA is matte, to keep glare out of the cameras.

The receiving box uses the same trick: marker ID 16 in its floor, dimensions identical to the simulator, because those millimetres are part of the "cube in box" success criterion.

## 2. Calibration from two cubes lying on the table

[03, video autocalib-real.mp4] The cubes just lie there. The arm tours 24 poses per layout, pointing the wrist camera at the cubes, while the top camera watches the same cubes the whole time. A single optimisation over marker-corner pixels from both cameras then solves for everything at once: the top camera's pose relative to the robot, the wrist camera's pose relative to the gripper, per-joint corrections, and the positions of the cubes themselves.

Final solution: two layouts, 48 poses, 101 observations, residual 1.6 px. On 12 held-out poses not used in the fit, the cube position agrees to 2.9 mm (median). The manual camera-to-robot calibration we had before was off by 22 mm and 2.6°.

The most important finding: without joint corrections the residual is 15 px instead of 1.6. The real servos differ from the model by up to 14 % in scale and 5.5° in offset. The cubes saw it; a person with a ruler never would.

We applied the same idea to a mobile robot: two cardboard cubes at the dock and phone cameras brought the pose error from 17 cm down to 0.6 cm.

## 3. First grasp with no human in the loop

[04, video grasp-real-first.mp4] Using that calibration, the arm for the first time found a cube with the top camera on its own, approached, descended, closed the fingers across the faces, lifted, checked from above that the spot on the table was empty, and put the cube back 17 mm from where it started. No interventions, no safety trips.

The first attempt ran into the software floor: under load the arm sags by 4 mm. We lowered the threshold and raised the grasp target by 6 mm. You only learn these things on hardware.

## 4. The robot records its own dataset

[05, video dataset-preview.mp4] The same cycle became a data collector: find the cube, grasp, lift, confirm with the wrist camera that the cube is in the fingers, carry it to a random point in the zone, release, return. Both cameras and all joints are recorded every tick at 30 Hz.

Two tricks make the human unnecessary. First, the cubes are taken in turns, and every drop to a random point reshuffles the scene. Second, wrist-camera refinement: the top camera at half a metre is off by up to 3 cm (marker seen from the side, shadows), so 3 cm above the cube the arm re-measures it with the wrist camera and descends on the refined coordinates. The two cameras agree within 5 mm.

Pilot run: 22 episodes, 19 successful, cube placed at a random target with a median error of 17.6 mm. Then, in one evening, the robot recorded 100 episodes, 42 thousand frames, 16 GB of raw data, while we did other things.

## 5. Recolouring: black-and-white cubes become coloured

[06, 07, video recolor-preview.mp4] Markers are for the algorithms; a policy has to learn colours: "pick up the red cube". So the cubes are recoloured in post-processing, geometrically. The faces are projected into the frame from the known cube pose, black cells are replaced with the brightness of the white on the same face, and the final colour is the target colour multiplied by per-pixel brightness. Shadows, shading and edges survive. The orange gripper fingers over the cube stay fingers: the occluder is recognised by the hue of the arm itself.

One recording thus becomes a dataset in six colours. Orange is excluded from the palette: it is the arm's colour. Honestly: after four iterations some artefacts remain, at the cube's edge and around the wrist camera housing. That is an open task.

## 6. GR00T: simulator first, then hardware

We had already tuned the recipe for fine-tuning GR00T N1.7 on a single RTX 4090 in simulation: 86 minutes per model, peak memory 21.7 GB of 24, a record of 92.2 % success over 1200 attempts. The evaluation protocol is strict: a single batch of 100 attempts has ±2.3 pp of noise, so an improvement is only claimed on 1200 attempts across two independent seed sets. The recipe and three datasets are public (links at the end).

For hardware we narrowed the task: the policy only finds and lifts the cube, and a plain algorithm carries it to the box, whose pose is known from calibration. In simulation the policy on shortened episodes lifts 88 of 100, and combined with the algorithm 86 cubes out of 100 end up in the box [08, video sim-lift-carry.mp4]. Every miss is on the policy's side.

[09, video first-real-lift-by-policy.mp4] On real data the first model (100 episodes, 10k steps) scored 0 of 3: the arm approached correctly and descended to 4.5 cm, then froze there and closed the fingers next to the cube. Watching the videos gave the reason: every recorded episode contains 4 seconds of the arm settling motionless, and the behaviour clone learned "stand still". We cut the idle segments from the data, doubled the steps, and in the morning the policy lifted a real cube: 3 of 5 attempts on video, 1 of 5 by the strict criterion (cube marker seen by the wrist camera). Small numbers, but it is the first time a learned model, not a script, acts on the real arm.

## 7. What we learned along the way

- A neural network copies everything faithfully, pauses included. Static segments in demonstrations must be cut.
- Beautiful conclusions need re-checking. We were sure letter-boxing the image cost 20 pp until we completed the matrix: the difference was single-seed noise.
- Do not load the machine while the arm is working: 24 ffmpeg processes crashed the collector with a serial-port timeout.
- The rig has a life of its own: the camera drops off USB, a cube rolls out of the zone, someone turns the lights off. So we added a calibration watchdog that checks from the frames, during operation, that cameras, robot and box are still in place, and a self-collision guard based on the MuJoCo model.
- A night run in a dark room is meaningless: frame brightness 27 versus 130 in the dataset. The runner now refuses to start without light.

## What's next

Hundreds more collector episodes, training on them, a hybrid "policy approaches, algorithm centres and grasps", and testing the recoloured data on cubes without markers. Everything above was done on a desk with two webcams in two weeks.

Links: the GR00T-on-one-4090 recipe — github.com/VShirokun/gr00t-on-4090; datasets — huggingface.co/VShirokun (office-rover-miss-v21, grasp-hd-v21, 2task-raw, CC BY 4.0).

---

### Media captions

- [01] `01-cubes-on-table.jpg` — the two marker cubes on the table, top camera view.
- [02] `02-cube-from-wrist-cam.jpg` — a cube as seen by the wrist camera before descent.
- [03] `03-autocalib.jpg`, `autocalib-real.mp4` — the calibration tour: green = detected markers, red = model prediction.
- [04] `04-first-grasp.jpg`, `grasp-real-first.mp4` — the first autonomous grasp.
- [05] `05-dataset-preview.jpg`, `dataset-preview.mp4` — sped-up preview of dataset recording, both cameras.
- [06] `06-recolor-stills.jpg` — 23 episodes recoloured into six colours.
- [07] `07-recolor-v4-check.jpg` — recolouring check frames: raw, old and new version.
- [08] `08-sim-lift-carry.jpg`, `sim-lift-carry.mp4` — policy lifts, algorithm carries to the box (simulator).
- [09] `09-first-real-lift-by-policy.jpg`, `first-real-lift-by-policy.mp4` — the first real cube lifted by a neural network.
