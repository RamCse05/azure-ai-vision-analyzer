import os

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from dotenv import load_dotenv

from azure.cognitiveservices.vision.computervision import (
    ComputerVisionClient
)

from azure.cognitiveservices.vision.computervision.models import (
    VisualFeatureTypes
)

from msrest.authentication import (
    CognitiveServicesCredentials
)

# ---------------------------------------------------
# Flask App
# ---------------------------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------
# Azure AI Vision Configuration
# ---------------------------------------------------

load_dotenv()

cog_endpoint = os.getenv("COG_SERVICE_ENDPOINT")
cog_key = os.getenv("COG_SERVICE_KEY")

credential = CognitiveServicesCredentials(cog_key)

cv_client = ComputerVisionClient(
    cog_endpoint,
    credential
)

# ---------------------------------------------------
# Serve Uploaded Images
# ---------------------------------------------------

@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )

# ---------------------------------------------------
# Analyze Image Function
# ---------------------------------------------------

def analyze_image(image_path):

    results = {
        "caption": "",
        "tags": [],
        "objects": [],
        "image": ""
    }

    with open(image_path, "rb") as image_stream:

        features = [
            VisualFeatureTypes.description,
            VisualFeatureTypes.tags,
            VisualFeatureTypes.objects
        ]

        analysis = cv_client.analyze_image_in_stream(
            image_stream,
            features
        )

    # Caption
    if analysis.description.captions:

        results["caption"] = (
            analysis.description.captions[0].text
        )

    # Tags
    if analysis.tags:

        for tag in analysis.tags:

            results["tags"].append(tag.name)

    # Objects
    if analysis.objects:

        for detected_object in analysis.objects:

            results["objects"].append(
                detected_object.object_property
            )

    return results

# ---------------------------------------------------
# Main Route
# ---------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    analysis_result = None

    if request.method == "POST":

        if "image" not in request.files:

            return "No image uploaded"

        image = request.files["image"]

        if image.filename == "":

            return "No selected image"

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image.filename
        )

        image.save(image_path)

        analysis_result = analyze_image(image_path)

        analysis_result["image"] = (
            "/uploads/" + image.filename
        )

    return render_template(
        "index.html",
        result=analysis_result
    )

# ---------------------------------------------------
# Run Flask
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
