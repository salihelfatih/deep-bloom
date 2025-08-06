import torch
import gradio as gr
from torchvision import transforms
from fulla_core.model import create_fulla_model
from PIL import Image

# Loading model and class names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_fulla_model()
model.load_state_dict(torch.load("fulla_model.pth", map_location=device))
model.to(device)
model.eval()

# 🌼 Flower class names
class_names = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemony",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


# 🌷 Defining the prediction function
def predict(image):
    # Taking an image, transforming it, and returning the model's prediction
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        # Using softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        # Getting the top 5 predictions
        confidences = {class_names[i]: float(probabilities[i]) for i in range(102)}

    return confidences


# 🌸 Creating the Gradio Interface
if __name__ == "__main__":
    # Defining a floral theme for the Gradio interface
    floral_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.pink,
        secondary_hue=gr.themes.colors.rose,
        neutral_hue=gr.themes.colors.gray,
        font=(gr.themes.GoogleFont("DM Sans"), gr.themes.GoogleFont("Source Code Pro")),
    ).set(
        body_background_fill="linear-gradient(to right, #fde4e4, #fde4e4, #fde4e4, #e6e6fa, #e6e6fa)",
        body_background_fill_dark="linear-gradient(to right, #2c1a2b, #1a1a2b)",
    )
    # Using gr.Blocks for a more custom layout
    with gr.Blocks(theme=floral_theme) as interface:
        gr.Markdown("# 🌸 Fulla Flower Classifier 🌸")
        gr.Markdown(
            "Upload a picture of a flower and see what Fulla thinks it is! This model was trained on the Flowers102 dataset using PyTorch and Transfer Learning."
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload an Image")
            label_output = gr.Label(num_top_classes=5, label="Top 5 Predictions")

        predict_button = gr.Button("Classify Flower")
        predict_button.click(fn=predict, inputs=image_input, outputs=label_output)

        gr.Examples(
            examples=[
                "assets/image_01340.jpg",
                "assets/image_02550.jpg",
                "assets/image_04430.jpg",
                "assets/image_05020.jpg",
                "assets/image_05440.jpg",
                "assets/image_05466.jpg",
                "assets/image_05602.jpg",
                "assets/image_06500.jpg",
                "assets/image_06743.jpg",
                "assets/image_07142.jpg",
                "assets/image_07436.jpg",
                "assets/image_07777.jpg",
                "assets/image_07983.jpg",
            ],
            inputs=image_input,
            outputs=label_output,
            fn=predict,
        )

        gr.Markdown("--- \n*Developed by Salih Elfatih as a capstone project.*")

    # Launching the Gradio App
    interface.launch(share=True)
