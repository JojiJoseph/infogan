// Elements
var canvas = document.querySelector("#canvas")
var btnErase = document.querySelector('#btn-erase')
var btnPredict = document.querySelector('#btn-predict')
var btnOk = document.querySelector('#btn-ok')
var modalDialogContainer = document.querySelector('#modal-dialog-container')
var outputBox = document.querySelector("#output-box")
var ctx = canvas.getContext("2d")

btnOk.addEventListener("click", function(event){
  modalDialogContainer.classList.add("hidden")
})

ctx.fillStyle = "black"
ctx.strokeStyle = "white"
ctx.lineWidth = 15
ctx.fillRect(0,0,canvas.width, canvas.height)

btnErase.addEventListener("click", function(event){
  ctx.fillRect(0,0,canvas.width, canvas.height)
});

btnPredict.addEventListener("click", function(event){
  var newCanvas = new OffscreenCanvas(28, 28);
  newCanvas.getContext("2d").drawImage(canvas, 0, 0, canvas.width, canvas.height, 0,0, 28, 28)
  data = newCanvas.getContext("2d").getImageData(0,0,28,28).data
  fetch('', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ "data":Array.from(data), "model":"mnist"}),
  })
  .then((response) => response.json())
  .then((data) => {
    modalDialogContainer.classList.remove("hidden")
    outputBox.innerText = data.prediction
  })
  .catch((error) => {
    console.error('Error:', error);
  });
});
var down = false
var x,y

canvas.addEventListener("mousedown", function(event){
  down = true
  x = event.offsetX
  y = event.offsetY
  ctx.beginPath();
  ctx.moveTo(x,y)
})

canvas.addEventListener("mousemove", function(event){
  if(down) {
    ctx.lineTo(event.offsetX, event.offsetY)
    x = event.offsetX
    y = event.offsetY
    ctx.stroke()
  }
})

canvas.addEventListener("mouseup", function(event){
  if (down) {
    ctx.closePath()
  }
  down = false
})
