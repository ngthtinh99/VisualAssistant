# VisualAssistant
An application to assist the visual impaired.

## Information

Visual US Team at [Accessibility Design Competition 2023](https://www.accessibilitydesigncompetition.com/).

## How to run the deploy
1. Clone repository.
<pre>git clone https://github.com/ngthtinh99/VisualAssistant.git</pre>
2. Install Python (Python 3.7 - 3.9 is required for supporting Pytorch).
3. Install necessary libraries.
<pre>pip install requests torch torchvision gradio timm fairscale transformers</pre>
4. Run the deploy, the first time downloading the model would take about 5-10 minutes, the next time would not need to reload.
<pre>python app.py</pre>
5. Browse the deploy on Localhost via the link http://localhost:7860, or the Public link generated in Command prompt.
6. Enjoy 🙂

## References

[[1](https://github.com/salesforce/BLIP)] BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.

[[2](https://arxiv.org/abs/2201.12086)] Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi, BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.

[[3](https://github.com/gradio-app/gradio)] Gradio: Build Machine Learning Web Apps — in Python.

