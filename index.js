const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const { createCanvas, loadImage } = require('canvas');

async function loadImageAsTensor(path) {
  const image = await loadImage(path);
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.width, image.height);
  // ✅ Pass the entire canvas instead of imageData
  return tf.browser.fromPixels(canvas);
}

async function classifyImage(imagePath) {
  const imageTensor = await loadImageAsTensor(imagePath);
  const model = await mobilenet.load();
  const predictions = await model.classify(imageTensor);

  console.log('✅ Predictions:');
  predictions.forEach((p, i) =>
    console.log(`${i + 1}. ${p.className} (${(p.probability * 100).toFixed(2)}%)`)
  );
}

classifyImage('./lion.jpg'); // replace with your image file name
