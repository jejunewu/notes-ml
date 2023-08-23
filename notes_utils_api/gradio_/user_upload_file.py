import json
import gradio as gr
import os
import tempfile

tempfile.tempdir = "/aidata/junjie/tmp"

'''
上传一个txt文本，并且在线解析内容
'''


class Args:
    def __init__(self):
        self.path = ""


args = Args()



def upload_file(files):
    args.path = files.name
    files.delete = True
    print(args.path)
    return files.name


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
