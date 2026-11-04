/**
 * Interactive Drawing Canvas for MNIST Exhibition Demo.
 * Smooth anti-aliased white brush on black background with touch and mouse support.
 */

class DrawingCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.isDrawing = false;
    this.hasDrawn = false;
    this.lastX = 0;
    this.lastY = 0;
    this.brushSize = 18;

    this.initCanvas();
    this.bindEvents();
  }

  initCanvas() {
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = this.brushSize;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
  }

  bindEvents() {
    // Mouse events
    this.canvas.addEventListener('mousedown', (e) => this.startDraw(e));
    this.canvas.addEventListener('mousemove', (e) => this.draw(e));
    window.addEventListener('mouseup', () => this.stopDraw());

    // Touch events
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      this.startDraw(mouseEvent);
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      this.draw(mouseEvent);
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.stopDraw();
    }, { passive: false });
  }

  getPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  startDraw(e) {
    this.isDrawing = true;
    this.hasDrawn = true;
    const pos = this.getPos(e);
    this.lastX = pos.x;
    this.lastY = pos.y;
    this.drawPoint(pos.x, pos.y);
  }

  draw(e) {
    if (!this.isDrawing) return;
    const pos = this.getPos(e);
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(pos.x, pos.y);
    this.ctx.stroke();
    this.lastX = pos.x;
    this.lastY = pos.y;
  }

  drawPoint(x, y) {
    this.ctx.beginPath();
    this.ctx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fill();
  }

  stopDraw() {
    this.isDrawing = false;
  }

  clear() {
    this.initCanvas();
    this.hasDrawn = false;
  }

  loadImageFromArray(pixels784) {
    this.clear();
    const imgData = this.ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const val = pixels784[i];
      const idx = i * 4;
      imgData.data[idx] = val;     // R
      imgData.data[idx + 1] = val; // G
      imgData.data[idx + 2] = val; // B
      imgData.data[idx + 3] = 255; // A
    }

    // Draw scaled up to canvas size
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    tempCanvas.getContext('2d').putImageData(imgData, 0, 0);

    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(tempCanvas, 0, 0, this.canvas.width, this.canvas.height);
    this.hasDrawn = true;
  }

  loadImageFromDataURL(dataURL) {
    const img = new Image();
    img.onload = () => {
      this.clear();
      this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
      this.hasDrawn = true;
    };
    img.src = dataURL;
  }

  getImageDataURL() {
    return this.canvas.toDataURL('image/png');
  }
}

