from compel import Compel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import streamlit as st
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import json
import http
import base64
from io import BytesIO

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')


runway = "runwayml/stable-diffusion-v1-5"
fast_sdxl = "prodia/sdxl-stable-diffusion-xl"

DEVICE = "mps"
FLAG = False


def save_image(filename, image):
    image.save(filename)
    print("in save")


def clean_slate():
    set_flag_false()
    st.session_state.generated_image = None


def set_flag_true():
    global FLAG
    FLAG = True


def set_flag_false():
    global FLAG
    FLAG = False


def generate_prompt(generation_dict):
    if generation_dict["prompt"]:
        prompt = f'["{generation_dict["prompt"]} with these characteristics","{generation_dict["street"]["type"]}road", "{generation_dict["weather"]["type"]} weather","Indian {generation_dict["background"]["type"]}","{generation_dict["obstacles"]["type"]}","{generation_dict["congestion"]} traffic" ,"{generation_dict["time"]} time"].and(1,1,{generation_dict["street"]["intensity"]},{generation_dict["weather"]["intensity"]},{generation_dict["background"]["intensity"]},{generation_dict["obstacles"]["intensity"]},1,1)'
    else:
        prompt = f'["{generation_dict["street"]["type"]}road", "{generation_dict["weather"]["type"]} weather","Indian {generation_dict["background"]["type"]}","{generation_dict["obstacles"]["type"]}","{generation_dict["congestion"]} traffic" ,"{generation_dict["time"]} time"].and(1,{generation_dict["street"]["intensity"]},{generation_dict["weather"]["intensity"]},{generation_dict["background"]["intensity"]},{generation_dict["obstacles"]["intensity"]},1,1)'
    return prompt


def text_to_img(generation_dict):
    print(FLAG, "text model")

    sdxl_text = StableDiffusionPipeline.from_pretrained(
        runway, torch_dtype=torch.float16).to(DEVICE)
    sdxl_text.safety_checker = None
    sdxl_text.requires_safety_checker = False
    sdxl_text.load_lora_weights('pytorch_lora_weights.safetensors')

    compel = Compel(tokenizer=sdxl_text.tokenizer,
                    text_encoder=sdxl_text.text_encoder)
    final_prompt = generate_prompt(generation_dict)
    conditioning = compel(final_prompt)
    negative_prompt = "cartoon, satellite"
    negative_conditioning = compel(negative_prompt)

    # generate image
    image = sdxl_text(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
                      num_inference_steps=35, output_type="pil").images[0]

    return image


def img_to_img(generation_dict, image):
    print(FLAG, "image model")

    sdxl_image = StableDiffusionImg2ImgPipeline.from_pretrained(
        runway, torch_dtype=torch.float16).to(DEVICE)

    init_image = image.resize((768, 512))

    YOUR_GENERATED_SECRET = "xS3qCTxTtnuP9wViznTa:be115cea45bdb6edf8ea79dc67e77f4e255963185f0aeee316a5e79b6a45dcc4"
    im_file = BytesIO()
    image.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    encoded_image = base64.b64encode(im_bytes).decode("utf-8")
    data = {
        "data": [
            {"image": f"data:image/jpeg;base64,{encoded_image}", "features": []},
        ]
    }

    headers = {
        "x-api-key": f"token {YOUR_GENERATED_SECRET}",
        "content-type": "application/json",
    }

    connection = http.client.HTTPSConnection("api.scenex.jina.ai")
    connection.request("POST", "/v1/describe", json.dumps(data), headers)
    response = connection.getresponse()
    response_data = response.read().decode("utf-8")

    data = json.loads(response_data)
    print(data)
    text = data['result'][0]['text']
    connection.close()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    ignore_words = ["image", 'Image', 'game',
                    'cartoon', 'sketch', 'Cartoon', 'Sketch']

    tagged_tokens = pos_tag(tokens)

    meaningful = [word for word, pos in tagged_tokens if pos == "NN" and word.lower(
    ) not in ignore_words and word.lower() not in stop_words]

    character = " ".join(meaningful[:15 if len(
        meaningful) > 15 else len(meaningful)-1])

    print(character)

    # final_prompt = generate_prompt(generation_dict)
    final_prompt = "Generate a new photorealistic image of an indian road with sharp defined edges, with these characteristics " + character + " " + \
        f'{generation_dict["street"]["type"]}{generation_dict["street"]["intensity"]}road, {generation_dict["weather"]["type"]}{generation_dict["weather"]["intensity"]} weather,Indian {generation_dict["background"]["type"]}{generation_dict["background"]["intensity"]} background,{generation_dict["obstacles"]["type"]}{generation_dict["obstacles"]["intensity"]} obstacle,{generation_dict["congestion"]} ,{generation_dict["time"]} time'
    image = sdxl_image(prompt=final_prompt, negative_prompt="A game type scene, cartoon drawing, cartoon, game, satellite",
                       image=init_image,
                       strength=0.75, guidance_scale=7.5).images[0]

    return image


if __name__ == "__main__":
    st.title('Synthetic Dataset Generator')
    # generated_image session to track accross reruns
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
        print("updated session")

    print("session:", st.session_state.generated_image)
    print("FLAG:", FLAG)

    with st.form(key='my_form', clear_on_submit=True):

        reference_image = st.file_uploader("Upload reference image here", type=[
            "png", "jpg", "jpeg"], accept_multiple_files=False)

        prompt = st.text_input('Enter a flexible prompt:')

        st.text("Or, Use the weighted form for fine grained control.")

        street = st.columns(2)
        with street[0]:
            street_type = st.text_input('Street type:')
        with street[1]:
            street_wt = st.slider(
                'Street Weights: ', 0.0, 2.0, 1.0, 0.1)

        weather = st.columns(2)
        with weather[0]:
            weather_type = st.text_input('Weather type:')
        with weather[1]:
            weather_wt = st.slider(
                'Weather Weights: ', 0.0, 2.0, 1.0, 0.1)

        background = st.columns(2)
        with background[0]:
            background_type = st.text_input('Background / Scene:')
        with background[1]:
            background_wt = st.slider(
                'Background Weights: ', 0.0, 2.0, 1.0, 0.1)

        obstacles = st.columns(2)
        with obstacles[0]:
            obstacles_type = st.text_input('Obstacle type:')
        with obstacles[1]:
            obstacles_wt = st.slider(
                'Obstacle Weights: ', 0.0, 2.0, 1.0, 0.1)

        congestion = st.selectbox(
            'Congestion Level: ', ["None", "Low", "Medium", "High"])

        time = st.text_input('Time of Day', "Morning")

        submitted = st.form_submit_button(label='Submit')

        if submitted:

            generation_dict = {
                "prompt": prompt,
                "street": {
                    "type": street_type,
                    "intensity": street_wt
                },
                "weather": {
                    "type": weather_type,
                    "intensity": weather_wt
                },
                "background": {
                    "type": background_type,
                    "intensity": background_wt
                },
                "obstacles": {
                    "type": obstacles_type,
                    "intensity": obstacles_wt
                },
                "congestion": congestion,
                "time": time,
            }
            # if reference image is given; set flag to true
            # first or after a reset
            if reference_image != None:
                print("reference not none")
                set_flag_true()
                reference_image = Image.open(reference_image)

            # if an image is already generated, set reference to generated.
            # only if a new reference image is not given
            if st.session_state.generated_image != None and reference_image == None:
                print("session still active")
                set_flag_true()
                reference_image = st.session_state.generated_image

            if not FLAG and generation_dict:
                generated_image = text_to_img(generation_dict)
                st.session_state.generated_image = generated_image
                st.image(np.array(generated_image))
                set_flag_false()

            elif FLAG and generation_dict:
                generated_image = img_to_img(generation_dict, reference_image)
                st.session_state.generated_image = generated_image
                st.image(np.array(generated_image))
                set_flag_false()

    columns = st.columns(2)
    with columns[0]:
        st.button("Reset", on_click=clean_slate)
    if st.session_state.generated_image != None:
        with columns[1]:

            filename = st.text_input("Enter the filename: ")
            image = st.session_state.generated_image
            path = f"{filename}.jpg"
            print(type(image))
            save = st.button("Save", on_click=save_image,
                             args=(path, image))
