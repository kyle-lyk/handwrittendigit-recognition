// Define a global variable to store the canvas context (ctx)
let ctx;
let isDrawing = false;

// Function to initialize the canvas when the page first loads
function initCanvas() {
    const canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
}

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
    predictDigit(); // Call the predictDigit() function when the drawing stops
}

function draw(e) {
    if (!isDrawing) return;
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    // Red stroke
    ctx.strokeStyle = '#FF0000';
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getBoundingBox() {
    // Get the canvas pixel data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Initialize bounding box coordinates
    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = 0;
    let maxY = 0;

    // Find the bounding box of the drawn pixels
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const pixelIndex = (y * canvas.width + x) * 4;
            if (data[pixelIndex + 3] > 0) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    // Return the bounding box coordinates
    return { minX, minY, maxX, maxY };
}

function cropCanvas(boundingBox) {
    const { minX, minY, maxX, maxY } = boundingBox;

    // Calculate the dimensions of the bounding box
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;

    // Calculate the padding based on 2% of the bounding box dimensions
    const paddingX = Math.round(boxWidth * 0.10);
    const paddingY = Math.round(boxHeight * 0.10);

    // Calculate the padded bounding box coordinates
    const paddedMinX = Math.max(0, minX - paddingX);
    const paddedMinY = Math.max(0, minY - paddingY);
    const paddedMaxX = Math.min(canvas.width, maxX + paddingX);
    const paddedMaxY = Math.min(canvas.height, maxY + paddingY);
    const croppedWidth = paddedMaxX - paddedMinX + 1;
    const croppedHeight = paddedMaxY - paddedMinY + 1;

    // Create a new canvas with the size of the cropped region plus padding
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = croppedWidth;
    croppedCanvas.height = croppedHeight;

    // Get the cropped context
    const croppedCtx = croppedCanvas.getContext('2d');

    // Draw the extracted ROI onto the cropped canvas with padding
    croppedCtx.drawImage(canvas, paddedMinX, paddedMinY, croppedWidth, croppedHeight, 0, 0, croppedWidth, croppedHeight);

    return croppedCanvas;
}


function predictDigit() {
    // Find the bounding box of the drawn digit
    const boundingBox = getBoundingBox();

    // Get the ROI data
    const croppedCanvas = cropCanvas(boundingBox);

    // Convert the cropped canvas to base64 data URL
    const croppedImage = croppedCanvas.toDataURL();

    // DEBUG: Download the raw image
    // const rawImage = canvas.toDataURL();
    // var link = document.createElement('a');
    // link.download = 'image.png';
    // link.href = rawImage;
    // link.click();

    // DEBUG: Download the cropped image
    // var link = document.createElement('a');
    // link.download = 'cropped.png';
    // link.href = croppedImage;
    // link.click();

    $.ajax({
        type: 'POST',
        url: '/predict/',
        data: {
            imageBase64: croppedImage
        },
        success: function (response) {
            endresult = JSON.parse(JSON.stringify(response))
            console.log(endresult)

            $('#prediction-result').html('Prediction is: <span class="pred-text">' + endresult.prediction + '</span>')
            $('#probs-img').attr('src', 'data:image/png;base64, ' + endresult.probabilities)
            $('#interpret').prop('src', 'data:image/png;base64,' + endresult.interpretations)
        },
        error: function (error) {
            console.error('Prediction Error:', error);
        }
    });
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);

// Attach the clearCanvas function to the "Clear" button click event
document.getElementById('clear-btn').addEventListener('click', clearCanvas);

// Call the initCanvas function when the page loads
document.addEventListener('DOMContentLoaded', initCanvas);