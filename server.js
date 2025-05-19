import express from 'express';
import multer from 'multer';
import { createWorker } from 'tesseract.js';
import { createCanvas, loadImage } from 'canvas';
import fetch from 'node-fetch';
import fs from 'fs';
import cors from 'cors';

const app = express();
const upload = multer({ dest: 'uploads/' });
app.use(cors());

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const MODEL_VERSION = "95b722319f13c7ed2c9e624cb4c81b81b60446e92a4e66644c8b20c1b2ec6404";

app.post('/api/replace-text', upload.single('image'), async (req, res) => {
  try {
    const imageFile = req.file;
    const { originalText, newText } = req.body;

    if (!imageFile || !originalText || !newText) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const buffer = fs.readFileSync(imageFile.path);

    // OCR with Tesseract
    const worker = await createWorker('eng');
    const { data: { words } } = await worker.recognize(buffer);
    await worker.terminate();

    const match = words.find(w =>
      w.text.trim().toLowerCase().includes(originalText.trim().toLowerCase())
    );

    if (!match) {
      console.log("OCR words:", words.map(w => w.text));
      return res.status(404).json({ error: 'Text not found', ocr: words.map(w => w.text) });
    }

    // Create mask with padding
    const image = await loadImage(buffer);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    const padding = 20;
    const x0 = Math.max(0, match.bbox.x0 - padding);
    const y0 = Math.max(0, match.bbox.y0 - padding);
    const x1 = Math.min(image.width, match.bbox.x1 + padding);
    const y1 = Math.min(image.height, match.bbox.y1 + padding);

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);

    const imageBase64 = `data:${imageFile.mimetype};base64,${buffer.toString('base64')}`;
    const maskBuffer = canvas.toBuffer('image/png');
    const maskBase64 = `data:image/png;base64,${maskBuffer.toString('base64')}`;

    // Call Replicate API
    const replicateRes = await fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: {
        "Authorization": `Token ${REPLICATE_API_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        version: MODEL_VERSION,
        input: {
          image: imageBase64,
          mask: maskBase64,
          prompt: `Replace the text "${originalText}" with "${newText}" in the same font, size, and style.`
        },
      }),
    });

    const prediction = await replicateRes.json();
    console.log("Raw replicate response:", JSON.stringify(prediction, null, 2));

    if (!replicateRes.ok) {
      throw new Error("Replicate API error: " + JSON.stringify(prediction));
    }

    if (!prediction?.urls?.get) {
      throw new Error("Replicate response did not include a polling URL. Full response: " + JSON.stringify(prediction));
    }

    // Poll for completion
    const endpointUrl = prediction.urls.get;
    let result;
    for (let i = 0; i < 30; i++) {
      const poll = await fetch(endpointUrl, {
        headers: { Authorization: `Token ${REPLICATE_API_TOKEN}` },
      });
      const status = await poll.json();
      if (status.status === 'succeeded') {
        result = status.output[0];
        break;
      }
      if (status.status === 'failed') throw new Error("Processing failed");
      await new Promise(r => setTimeout(r, 1000));
    }

    fs.unlinkSync(imageFile.path);
    if (!result) return res.status(500).json({ error: 'Timeout' });

    res.json({ result });

  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
