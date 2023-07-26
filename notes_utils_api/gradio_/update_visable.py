import gradio as gr

species_map = {
    "Mammal": ["Elephant", "Giraffe", "Hamster"],
    "Fish": ["Shark", "Salmon", "Tuna"],
    "Bird": ["Chicken", "Eagle", "Hawk"],
}


def filter_species(species):
    return gr.Dropdown.update(
        choices=species_map[species], value=species_map[species][1]
    ), gr.update(visible=True)

def update_bttn():
    return gr.Dropdown.update(
        choices=species_map["Fish"], value=species_map["Fish"][1]
    ), gr.update(visible=True)


def filter_weight(animal):
    if animal in ("Elephant", "Shark", "Giraffe"):
        return gr.update(maximum=100)
    else:
        return gr.update(maximum=20)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Animal Generator
        """
    )
    # 种类按钮
    rad_species = gr.Radio(label="Animal Class", choices=["Mammal", "Fish", "Bird"])
    # 下拉动物
    dropdown_animal = gr.Dropdown(label="Animal", choices=[])
    button = gr.Button(value="点击转到Fish")


    with gr.Column(visible=False) as details_col:
        weight = gr.Slider(0, 20)
        details = gr.Textbox(label="Extra Details")
        generate_btn = gr.Button("Generate")
        output = gr.Textbox(label="Output")

    button.click(update_bttn, outputs=[dropdown_animal, details_col])

    rad_species.change(filter_species, inputs=rad_species, outputs=[dropdown_animal, details_col])

    dropdown_animal.change(filter_weight, dropdown_animal, weight)
    weight.change(lambda w: gr.update(lines=int(w / 10) + 1), weight, details)

    generate_btn.click(lambda x: x, details, output)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
