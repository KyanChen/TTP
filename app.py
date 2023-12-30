import gradio as gr
import glob
import torch
from opencd.apis import OpenCDInferencer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

config_file = 'configs/TTP/ttp_sam_large_levircd_infer.py'
checkpoint_file = 'ckpt/epoch_270.pth'

# build the model from a config file and a checkpoint file
mmcd_inferencer = OpenCDInferencer(
    model=config_file,
    weights=checkpoint_file,
    classes=['unchanged', 'changed'],
    palette=[[0, 0, 0], [255, 255, 255]],
    device=device
)

def infer(img1, img2):
    # test a single image
    result = mmcd_inferencer([[img1, img2]], show=False, return_vis=True)
    visualization = result['visualization']
    return visualization


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # [Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection](https://arxiv.org/abs/2312.16202)
        """)
    # a empty row
    gr.Row()

    with gr.Row():
        input_0 = gr.Image(label='Input Image1')
        input_1 = gr.Image(label='Input Image2')
    with gr.Row():
        output_gt = gr.Image(label='Predicted Mask')
    btn = gr.Button("Detect")
    btn.click(infer, inputs=[input_0, input_1], outputs=[output_gt])

    img1_files = glob.glob('samples/A/*.png')
    img2_files = [f.replace('A', 'B') for f in img1_files]
    input_files = [[x, y] for x, y in zip(img1_files, img2_files)]
    gr.Examples(input_files, fn=infer, inputs=[input_0, input_1], outputs=[output_gt], cache_examples=False)
    gr.Markdown(
        """
			This is the demo of ["Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection"](https://arxiv.org/abs/2312.16202). Seeing [Github](https://github.com/KyanChen/TTP) for more information!
		""")
    # a empty row
    gr.Row()

if __name__ == "__main__":
    demo.launch()