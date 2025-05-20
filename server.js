import express from 'express';
import multer from 'multer';
import { createWorker } from 'tesseract.js';
import { createCanvas, loadImage } from 'canvas';
import fetch from 'node-fetch';
import fs from 'fs';
import cors from 'cors';
import path from 'path';

const app = express();
const upload = multer({ dest: 'uploads/' });
app.use(cors());

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const MODEL_VERSION = "95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3";

app.post('/api/replace-text', upload.single('image'), async (req, res) => {
  try {
    const imageFile = req.file;
    const { originalText, newText } = req.body;

    if (!imageFile || !originalText || !newText) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const buffer = fs.readFileSync(imageFile.path);

    // OCR: detect text
    const worker = await createWorker('eng');
    const { data: { words } } = await worker.recognize(buffer);
    await worker.terminate();

    const match = words.find(w =>
      w.text.trim().toLowerCase().includes(originalText.trim().toLowerCase())
    );

    if (!match) {
      return res.status(404).json({ error: 'Text not found', ocr: words.map(w => w.text) });
    }

    // Create mask
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

    // Step 1: remove original text
    const step1Res = await fetch("https://api.replicate.com/v1/predictions", {
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
          prompt: "Remove any text in the masked area and leave it completely blank with background only.",
        },
      }),
    });

    const step1 = await step1Res.json();
    if (!step1Res.ok || !step1?.urls?.get) {
      return res.status(500).json({ error: 'Step 1 failed', details: step1 });
    }

    // Poll for step 1
    let cleanedImage;
    for (let i = 0; i < 30; i++) {
      const poll = await fetch(step1.urls.get, {
        headers: { Authorization: `Token ${REPLICATE_API_TOKEN}` },
      });
      const status = await poll.json();
      if (status.status === 'succeeded') {
        cleanedImage = status.output?.[0];
        break;
      }
      if (status.status === 'failed') {
        return res.status(500).json({ error: 'Step 1 polling failed' });
      }
      await new Promise(r => setTimeout(r, 1000));
    }

    if (!cleanedImage) return res.status(500).json({ error: 'Step 1 timed out' });

    // Step 2: add new text
    const step2Res = await fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: {
        "Authorization": `Token ${REPLICATE_API_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        version: MODEL_VERSION,
        input: {
          image: cleanedImage,
          mask: maskBase64,
          prompt: `Add the word "${newText}" in bold, uppercase blue futuristic font. Align it with the design and match the original style.`,
        },
      }),
    });

    const step2 = await step2Res.json();
    if (!step2Res.ok || !step2?.urls?.get) {
      return res.status(500).json({ error: 'Step 2 failed', details: step2 });
    }

    // Poll for step 2
    let finalImage;
    for (let i = 0; i < 30; i++) {
      const poll = await fetch(step2.urls.get, {
        headers: { Authorization: `Token ${REPLICATE_API_TOKEN}` },
      });
      const status = await poll.json();
      if (status.status === 'succeeded') {
        finalImage = status.output?.[0];
        break;
      }
      if (status.status === 'failed') {
        return res.status(500).json({ error: 'Step 2 polling failed' });
      }
      await new Promise(r => setTimeout(r, 1000));
    }

    fs.unlinkSync(imageFile.path);
    if (!finalImage) return res.status(500).json({ error: 'Step 2 timed out' });

    res.json({ result: finalImage });

  } catch (err) {
    console.error('Server error:', err);
    res.status(500).json({ error: err instanceof Error ? err.message : 'Unknown error' });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`✅ Server listening on port ${PORT}`));
