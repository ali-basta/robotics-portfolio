# Voice-Controlled Robotic Sorting System (LLM + Computer Vision + UR Robot Arm)

## Overview
A collaborative cobot system that lets a human give **natural-language voice commands, in any language**, to a UR robot arm to sort coins of different sizes and colors into the correct boxes. Built as part of research into collaborative robots (cobots) that can work safely and intuitively alongside human factory workers, without requiring the operator to learn any programming or fixed command syntax.

> Example command: *"Put 3 small green coins in the green box, then take 2 big orange coins and 4 blue coins of any size from left to right and put them all in the red box."*

The system understood and executed commands like this correctly.

## How it works

**1. Voice → Intent (LLM)**
The spoken instruction is captured and sent, together with a task-specific prompt, to an LLM. The prompt explains the sorting task and the available objects/boxes. The LLM parses the free-form human instruction and returns a structured JSON describing exactly which objects to pick (by color, size, and/or order), in what sequence, and into which destination box.

The system supports both a cloud LLM (OpenAI API) and a locally hosted LLM via Ollama, selectable through an environment variable. Both produced reliable structured output for this task; running the model locally via Ollama avoided ongoing per-request API costs, so it became the preferred setup once validated.

**2. Perception (Computer Vision)**
A camera continuously tracks the table using OpenCV, detecting the position, color, and size of every coin and box within a defined work area. All positions are expressed relative to shared reference points.

**3. Coordinate Unification**
The robot's native coordinate system was recalibrated to match the camera's coordinate system via a custom coordinate transformation. This was a deliberate design choice to simplify the math and make robot movements more precise and reliable, rather than constantly converting between two different reference frames at runtime.

**4. Execution (Robot Control)**
A server connected to the robot receives the matched object positions (resolved from the LLM's JSON output against the live camera detections) as HTTP requests. The robot then executes a full pick-and-place sequence starting from a home position: move to each target coin, pick it up, move to the correct box, place it, return to a shared ready pose, then repeat for the next object until the batch is complete.

**5. Quality Gate (Standalone Add-on)**
A separate, independent inspection service was built as an extension to the project: after a coin is placed on a conveyor belt, it passes through a 3D-printed gate equipped with a Raspberry Pi camera, which uses the same color/shape detection approach to confirm the correct coin was picked. This runs as its own Flask service and is not wired into the main voice/LLM pipeline above; it was developed as a standalone proof of concept for an inspection step that could sit further down the line.

## My contribution
This was a collaborative project built together with a colleague. I was involved across the full stack, including:
- Writing the robot control and camera/vision code, their respective servers, and the LLM/API integration
- Designing and implementing the coordinate transformation between the robot and camera reference frames
- Physical setup of the workstation (camera, lighting, conveyor, gate)
- Designing and 3D-printing the mechanical parts, including the quality-gate housing

## Tech stack
- **LLM:** Natural-language command parsing into structured JSON, with a swappable backend (OpenAI API or a locally hosted model via Ollama) selected via environment variable
- **Computer Vision:** Python, OpenCV (object detection, color/size classification, coordinate tracking)
- **Robot:** Universal Robots (UR) arm, custom server for movement control via HTTP
- **Hardware:** Raspberry Pi camera, 3D-printed quality-gate housing, conveyor belt
- **Architecture:** Flask-based HTTP services connecting perception, decision (LLM), and execution layers

## Files in this repo
| File | Purpose |
|---|---|
| `robot_job_manager.py` | Flask server: manages and executes queued pick-and-place jobs on the robot |
| `voice_vision_robot_bridge.py` | Client/orchestrator: voice capture → LLM parsing → vision matching → sends pick/place requests to the robot server |
| `ollama_client.py` | Wrapper for the locally hosted LLM (Ollama) backend; used as the primary LLM in practice to avoid ongoing API costs |
| `quality_gate.py` | Standalone Flask service: camera-based inspection for the conveyor quality gate |

## Demo
A short clip showing the full voice-to-pick-and-place cycle in action (voice command given in Arabic, captioned in English):

[**▶ Watch the demo video**](https://www.youtube.com/shorts/WAYct-D2C68)

---
*Note: this repository contains a showcase version of code originally developed as part of a university research project, shared with permission. Internal network addresses and credentials have been removed/replaced with environment variables.*
