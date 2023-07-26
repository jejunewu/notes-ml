import gradio as gr

dropdown_map = {
    "A": ["Elephant", "Giraffe", "Hamster"],
    "B": ["Shark", "Salmon", "Tuna"],
    "C": ["Chicken", "Eagle", "Hawk"],
}


def update_dropdown_by_radio(radio):
    return gr.Dropdown.update(
        choices=dropdown_map[radio], value=dropdown_map[radio][1]
    )


def update_textbox_by_dropdown(dropdown):
    return f"你选择了: {dropdown} !"


def update_dropdown_by_button(button):
    return gr.Dropdown.update(choices=["你", "我", "他"], value="你")


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # 通过按钮改变下拉框
    """
    )

    radio = gr.Radio(label="按钮选择", choices=["A", "B", "C"])
    button = gr.Button(value="点击选择")
    dropdown = gr.Dropdown(label="下拉选项", choices=[])
    textbox = gr.Textbox(label="输出信息")

    radio.change(update_dropdown_by_radio, radio, dropdown)
    button.click(update_dropdown_by_button, button, dropdown)
    dropdown.change(update_textbox_by_dropdown, dropdown, textbox)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
