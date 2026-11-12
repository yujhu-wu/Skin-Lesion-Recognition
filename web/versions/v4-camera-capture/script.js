document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const captureButton = document.getElementById('captureButton');
    const cameraPreview = document.getElementById('cameraPreview');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const snapButton = document.getElementById('snapButton');
    const retakeButton = document.getElementById('retakeButton');
    const uploadButton = document.getElementById('uploadButton');

    let stream;

    // Toggle camera preview and file input
    captureButton.addEventListener('click', () => {
        fileInput.style.display = 'none'; // Hide file input
        captureButton.classList.add('hidden'); // Hide Capture Photo button
        cameraPreview.classList.remove('hidden'); // Show camera preview
        startCamera();
    });

    // Start the camera
    function startCamera() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((mediaStream) => {
                stream = mediaStream;
                video.srcObject = mediaStream;
                video.classList.remove('hidden');
                canvas.classList.add('hidden');
                snapButton.classList.remove('hidden');
                retakeButton.classList.add('hidden');
            })
            .catch((err) => {
                console.error("Error accessing camera: ", err);
            });
    }

    // Take photo
    snapButton.addEventListener('click', () => {
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        video.classList.add('hidden'); // Hide video preview
        canvas.classList.remove('hidden'); // Show captured image on canvas
        snapButton.classList.add('hidden'); // Hide Take Photo button
        retakeButton.classList.remove('hidden'); // Show Retake Photo button
        stopCamera();
    });

    // Retake photo
    retakeButton.addEventListener('click', () => {
        video.classList.remove('hidden'); // Show video preview
        canvas.classList.add('hidden'); // Hide captured image on canvas
        snapButton.classList.remove('hidden'); // Show Take Photo button
        retakeButton.classList.add('hidden'); // Hide Retake Photo button
        startCamera(); // Restart the camera
    });

    // Stop the camera
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    }

    // Upload the image
    uploadButton.addEventListener('click', () => {
        const context = canvas.getContext('2d');
        canvas.toBlob((blob) => {
            const file = new File([blob], 'captured-image.png', { type: 'image/png' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            fileInput.style.display = 'block'; // Show file input again
        }, 'image/png');
    });
});
