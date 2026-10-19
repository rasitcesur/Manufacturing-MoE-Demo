"""Manufacturing-MoE Normal model demo. Weights are downloaded only after training."""
import json
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import torch
import spaces
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
MODEL_REPO = os.environ.get("MANUFACTURING_MOE_REPO", "rasitcesur/Manufacturing-MoE")
CHECKPOINTS = {
    "manufacturing": "checkpoints/manufacturing/model.pt",
    "robotics": "checkpoints/robotics/robotics_model.pt",
    "gmaw": "checkpoints/gmaw/arc_welding_model.pt",
    "rsw": "checkpoints/rsw/model.pt",
    "tig": "checkpoints/tig/model.pt",
}
PROCESSES = {
    "FDM 3D printing": ("fdm", "manufacturing", "Extrusion and travel moves; print time, power draw and filament consumed."),
    "CNC milling": ("milling", "manufacturing", "Spindle and feed motion; machining time, electricity and removed material."),
    "CNC turning": ("turning", "manufacturing", "Turning cuts and machine drives; processing time, energy and material removed."),
    "Drilling": ("drilling", "manufacturing", "Hole drilling, thrust-related loads, spindle and feed motor demand."),
    "Boring": ("boring", "manufacturing", "Hole enlargement and finish boring with its own cutting conditions."),
    "Reaming": ("reaming", "manufacturing", "Finishing a drilled hole with small stock removal."),
    "Tapping": ("tapping", "manufacturing", "Thread cutting with synchronized feed and spindle reversal."),
    "Grinding": ("grinding", "manufacturing", "Wheel and traverse demand in a surface grinding program."),
    "Band sawing": ("sawing", "manufacturing", "Saw blade drive, feed and kerf material removal."),
    "Laser cutting": ("laser", "manufacturing", "Beam on-time, machine travel, electrical power and kerf removal."),
    "Plasma cutting": ("plasma", "manufacturing", "Torch motion, plasma power, supporting equipment and kerf removal."),
    "General robotics": ("robotics", "robotics", "One to fifteen axes; time and electrical demand from joint dynamics and previous measurements."),
    "MIG/MAG (GMAW)": ("gmaw", "gmaw", "Welding arm motion, arc electricity, consumable wire, shielding gas and auxiliary loads."),
    "Resistance spot welding": ("rsw", "rsw", "Electrode pressure and current pulses, robot motion and electrical losses."),
    "TIG (GTAW)": ("tig", "tig", "Torch travel, arc electricity, shielding gas and optional filler metal."),
}
MACHINING = set(list(PROCESSES)[:11])


def checkpoint_path(family):
    path = CHECKPOINTS[family]
    local = os.environ.get("MANUFACTURING_MOE_CHECKPOINTS")
    if local and (Path(local) / path).is_file():
        return str(Path(local) / path)
    return hf_hub_download(repo_id=MODEL_REPO, filename=path, repo_type="model")


def status():
    try:
        remote = HfApi().list_repo_files(MODEL_REPO, repo_type="model")
        available = [f for f in CHECKPOINTS.values() if f in remote]
    except Exception as exc:
        local = os.environ.get("MANUFACTURING_MOE_CHECKPOINTS")
        available = [f for f in CHECKPOINTS.values() if local and (Path(local)/f).is_file()]
        if not available:
            error_type = type(exc).__name__
            print(f"Checkpoint status failed for {MODEL_REPO}: {error_type}: {exc}", flush=True)
            return (
                f"Cannot inspect `{MODEL_REPO}`. "
                f"Error type: `{error_type}`. "
                "See the Space runtime logs; this does not necessarily mean the checkpoints are missing."
            )
    items = []
    for family, name in CHECKPOINTS.items():
        items.append(f"{'✓' if name in available else '—'} **{family.upper()}**: {'trained checkpoint available' if name in available else 'training or upload pending'}")
    return "\n".join(items) + "\n\nThe demo never uses random weights for public predictions."


def examples(process_name):
    op, family, desc = PROCESSES[process_name]
    if family == "manufacturing":
        code = (ROOT / 'examples' / f'{op}.gcode').read_text()
        profile = (ROOT / 'examples' / f'{op}.json').read_text()
        return desc, gr.update(visible=True), gr.update(visible=False), code, profile, ""
    file_name = 'robotics_input.json' if family == 'robotics' else f'{family}_input.json'
    contents = (ROOT / 'examples' / file_name).read_text()
    return desc, gr.update(visible=False), gr.update(visible=True), "", "", contents


@spaces.GPU(duration=60)
def calculate(process_name, gcode, profile_text, robot_json):
    operation, family, _ = PROCESSES[process_name]
    try:
        ckpt = checkpoint_path(family)
    except Exception as exc:
        error_type = type(exc).__name__
        error_detail = str(exc)
        print(
            f"Checkpoint load failed: repository={MODEL_REPO}; "
            f"path={CHECKPOINTS[family]}; error={error_type}: {error_detail}",
            flush=True,
        )
        return {}, (
            f"**Could not load {family.upper()} checkpoint.** "
            f"Repository: `{MODEL_REPO}`. "
            f"File: `{CHECKPOINTS[family]}`. "
            f"Error type: `{error_type}`. "
            "See the Space runtime logs for the complete error. "
            "No prediction was generated."
        )
    try:
        if family == "manufacturing":
            if len(gcode) > 100_000 or len(profile_text) > 100_000:
                raise ValueError("The program or profile is too large for the online demo.")
            json.loads(profile_text)
            with tempfile.TemporaryDirectory() as work:
                gf=Path(work)/'program.ngc';pf=Path(work)/'profile.json'
                gf.write_text(gcode,encoding='utf8');pf.write_text(profile_text,encoding='utf8')
                from predict import predict
                mode = 'drilling' if operation in {'boring','reaming','tapping'} else operation
                result, _, _ = predict(ckpt,str(gf),str(pf),mode,operation if mode=='drilling' else None,device='auto')
        else:
            if len(robot_json) > 750_000:raise ValueError("The JSON input is too large for the online demo.")
            payload=json.loads(robot_json)
            if family=='robotics':
                from predict_robotics import robot_predict
                result=robot_predict(ckpt,payload['robot'],payload['motion'])
            elif family=='gmaw':
                from predict_arc_welding import infer
                result=infer(payload,ckpt)
            else:
                from predict_welding import calculate as welding_predict
                if payload.get('process')!=family:
                    raise ValueError(f"Process must be {family!r}.")
                result=welding_predict(payload,ckpt)
        return result, "**Result:** Synthetic-model estimate, not validated for production control or weld acceptance."
    except Exception as exc:
        return {}, f"Prediction could not be completed: `{type(exc).__name__}: {exc}`"


with gr.Blocks(title="Manufacturing MoE Demo") as demo:
    gr.Markdown("# Manufacturing MoE")
    gr.Markdown("Processing time, electricity and material estimates for 15 manufacturing operations. "
                "Each operation uses its own process physics; the robotics and welding models have separate trained weights.")
    with gr.Accordion("Model availability", open=False):
        available = gr.Markdown("Check whether the five trained branches have been published.")
        check_button=gr.Button("Check trained models", size="sm")
        check_button.click(status,outputs=available)
    select=gr.Dropdown(choices=list(PROCESSES),value="CNC milling",label="Operation")
    description=gr.Markdown(PROCESSES["CNC milling"][2])
    with gr.Group(visible=True) as machine_inputs:
        gcode=gr.Textbox(label="G-code program",lines=10)
        profile=gr.Textbox(label="Machine profile (JSON)",lines=11)
    with gr.Group(visible=False) as robot_inputs:
        process_json=gr.Textbox(label="Robot and process program (JSON)",lines=20)
        gr.Markdown("The robot and welding models require measurements from earlier cycles. The example JSON shows the expected structure.")
    run=gr.Button("Estimate",variant="primary")
    report=gr.Markdown()
    result=gr.JSON(label="Estimated outputs and component energies")
    gr.Markdown("The bundled examples come from synthetic test programs and do not establish real machine performance. "
                "**License:** [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). "
                "[Model and training code](https://huggingface.co/rasitcesur/Manufacturing-MoE).")
    select.change(examples,[select],[description,machine_inputs,robot_inputs,gcode,profile,process_json])
    run.click(calculate,[select,gcode,profile,process_json],[result,report])
    demo.load(examples,[select],[description,machine_inputs,robot_inputs,gcode,profile,process_json])

if __name__=="__main__":
    demo.launch()
