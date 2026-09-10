# Contest entry: NVIDIA GTC Berlin Golden Ticket

Deadline: **September 10, 2026** (entry period ends today; time zone not stated — post as early as possible).
Rules: post on LinkedIn, X or Instagram, hashtag `#NVIDIAGTC`, tag the judge you heard about the challenge from.
Judging (1–10 each, equal weight): technical innovation · effective use of NVIDIA/partner technology · impact for developers · quality of documentation and presentation.
Eligibility: individuals 18+, residents of the listed countries (Russia is NOT on the list — check your residence before posting).

## LinkedIn post (main entry)

Attach: `docs/contest/two-cubes-reel.mp4` (62 s, 1280×720). Tag **Asier Arranz** (NVIDIA, Robotics & Physical AI) via @-mention; optionally also Johnny Nunez (NVIDIA AI DevRel).

---

Two 3D-printed cubes with ArUco markers taught my real SO-101 arm to calibrate itself, record its own dataset, and learn to pick with NVIDIA Isaac GR00T N1.7 — on a single RTX 4090, no teleoperation, no motion capture. #NVIDIAGTC

What the two cubes do:
🔹 Self-calibration. The cubes just lie on the table. The arm tours 48 poses looking at them with a wrist camera while a top camera watches the same cubes; one optimisation over marker corners solves the top camera pose, the wrist camera pose, per-joint corrections and the cube positions. Held-out error 2.9 mm. The real servos turned out to be off by up to 14 % in scale — the cubes saw it, a ruler never would.
🔹 Autonomous data collection. Find → grasp → lift → confirm with the wrist camera → carry to a random spot → release. 200+ episodes recorded by the robot alone, both cameras at 30 Hz.
🔹 Geometric recolouring. Markers are for the algorithms; the policy needs colours. One recording becomes a dataset in any cube colours, fingers and shadows preserved.
🔹 Learning. GR00T N1.7 fine-tuned on one 24 GB RTX 4090 (86 min per model). Task split: the policy finds and lifts, a plain algorithm carries to the box — 86/100 in simulation, and today the fine-tuned model lifted a real cube for the first time.

Honest numbers, failures included: every miss is in the repo as a video, and the eval protocol needs 1200 attempts before we call anything an improvement.

Open source:
▪ github.com/VShirokun/so101-two-cubes — the cubes (print files), calibration, collector, recolouring, watchdog, self-collision guard, GR00T lift recipe
▪ github.com/VShirokun/gr00t-on-4090 — how to fine-tune GR00T N1.7 on one 24 GB GPU without root, three CC-BY datasets on Hugging Face, 1800 evaluated attempts as JSON

Stack: NVIDIA Isaac GR00T N1.7 · Cosmos-Reason2 backbone · MuJoCo · OpenCV ArUco · SO-101 (LeRobot) · RTX 4090

@Asier Arranz — entering the GTC Berlin Golden Ticket challenge with this one. #NVIDIAGTC #GR00T #PhysicalAI #Robotics #OpenSource

---

## X / Twitter (shorter, ≤280 chars per post; attach the same video)

Two marker cubes taught my real SO-101 arm to self-calibrate (2.9 mm), record its own dataset (200+ episodes, no teleop) and learn to pick with @NVIDIA Isaac GR00T N1.7 on one RTX 4090. Open source ↓ #NVIDIAGTC @asierarranz
github.com/VShirokun/so101-two-cubes · github.com/VShirokun/gr00t-on-4090

## Checklist before posting

1. Confirm residence eligibility (rules §2).
2. Make `so101-two-cubes` public on GitHub (the rules require an open-source application). Push from this machine or your Mac; the repo is prepared in `/srv/data/vshirokun/so101-two-cubes`.
3. Upload the video natively on LinkedIn (not a link), add the text above, @-mention the judge so the mention resolves.
4. Post from your personal account (individual entries only).
5. Keep the LinkedIn article as a follow-up: publish it as an article and link it in a comment under the post (`docs/article/linkedin-two-cubes-en.md`).
6. Save the post URL; NVIDIA contacts winners by DM/email within 48 h after September 14.
