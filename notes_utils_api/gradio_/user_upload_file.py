import gradio as gr
import json

import gradio as gr


class Args:
    def __init__(self):
        self.path = ""


args = Args()


def upload_file(files):
    print(files)
    files.seek(0)
    print(files.read())
    # file_paths = [file.name for file in files]
    args.path = files
    print(args.path)
    return files.read()


def read_text():
    with open(args.path) as f:
        data = f.readlines()
    return data


with gr.Blocks() as app:
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=["text"], file_count="single")
    upload_button.upload(upload_file, upload_button, file_output)
    btn_read = gr.Button(value='读取文本')
    text_res = gr.Textbox(label="生成信息")
    btn_read.click(fn=read_text, inputs=[], outputs=[text_res])

app.launch(server_name='0.0.0.0')
