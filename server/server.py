"""
Server to demonstrate PyTorch implementation
"""

from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import webbrowser
import sys
import numpy as np
import torch
import torchvision.transforms.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

# Changing current working directory to the directory of source code
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# from caps_net import CapsNet
# # from cifar_net import CifarNet
# from cifar_net import CifarNet
from mnist_infogan import Discriminator


model = Discriminator()
model.load_state_dict(torch.load(os.path.join(dname, '../mnist.pt'), map_location=torch.device('cpu')))
model.eval()

# cifar_model = CifarNet()
# cifar_model.load_state_dict(torch.load(os.path.join(dname, '../src/model_cifar'), map_location=torch.device('cpu')))
# cifar_model.load_state_dict(torch.load(os.path.join(dname, '../models/model_cifar'), map_location=torch.device('cpu')))
# cifar_model.eval()

def preprocess(data):
    """
    Preprocess image data from frontend.
    Given image data is of size 28*28*4. We remove 3 channels
    and make the data is of size, 1*28*28. Then reshape it to (1, 28, 28) torch.Tensor

    returns Tensor of shape (1, 28, 28)
    """
    data = np.array(data)
    # Get indices for G, B and A channels
    indices = [[i*4+1, i*4+2, i*4+3] for i in range(28*28*1)]
    indices = np.array(indices).reshape(1, -1)
    # Remove these unnecessary indices
    data = np.delete(data, indices).reshape(1, 28, 28)
    return data

# def preprocess_cifar(data):
#     """
#     Preprocess image data from frontend.
#     Given image data is of size 32*32*4. We remove 1 channel.
#     and make the data is of size, 1*32*32. Then reshape it to (1, 32, 32) torch.Tensor

#     returns Tensor of shape (1, 32, 32)
#     """
#     data = np.array(data)
#     # Get indices for G, B and A channels
#     indices = [[i*4+3] for i in range(32*32*1)]
#     indices = np.array(indices).reshape(1, -1)
#     # Remove these unnecessary indices
#     data = np.delete(data, indices).reshape(3, 32, 32)
#     return data

def get_prediction(data):
    """
    Return the predicted digit using preprocessed data
    data is a torch.Tensor with shape (1, 28, 28)
    """
    image = F.to_tensor(np.array(data).reshape(28,28,1)/255).float()
    # image = F.normalize(image,(0.1307,), (0.3081,))
    to_predict = torch.stack([image], dim = 0)
    # out, _, _ = model(to_predict)
    # out = out.squeeze(-1)
    # out = out ** 2
    # out = out.sum(dim=2)
    # out = out.view(10)
    _, out, _ = model(to_predict)
    prediction = np.argmax(out.data.numpy().squeeze())
    return prediction

# def get_prediction_cifar(data):
#     """
#     Return the predicted image using preprocessed data
#     data is a torch.Tensor with shape (1, 32, 32)
#     """
#     image = F.to_tensor(np.array(data).reshape(32, 32, 3)/255).float()
#     # image = F.normalize(image,(0.1307,), (0.3081,))
#     to_predict = torch.stack([image], dim = 0)
#     out, _, _ = cifar_model(to_predict)
#     out = out.squeeze(-1)
#     out = out ** 2
#     out = out.sum(dim=2)
#     out = out.view(10)
#     prediction = np.argmax(out.data.numpy().squeeze())
#     return prediction

class DemoServerHandler(SimpleHTTPRequestHandler):

    def send_json(self, json_message, status=HTTPStatus.OK):
        """
        Send json_message to frontend.
        json_message is a dictionary
        """
        encoded = json.dumps(json_message).encode("utf-8", "replace")
        length = len(encoded)

        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-length", length)
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        length = int(self.headers.get('content-length'))
        request = json.loads(self.rfile.read(length))

        if "data" in request:
            if request["model"] == "cifar":
                # data = request["data"]
                # data = preprocess_cifar(data)
                # prediction = get_prediction_cifar(data)
                # response = {"prediction": int(prediction)}
                # self.send_json(response)
                pass
            else:
                data = request["data"]
                data = preprocess(data)
                prediction = get_prediction(data)
                response = {"prediction": int(prediction)}
                self.send_json(response)
        else:
            response = {"error":"Not Implemented"}
            self.send_json(response, HTTPStatus.NOT_IMPLEMENTED)

def run(server_class=HTTPServer, handler_class=DemoServerHandler):
    """
    Run server on port 8000
    """
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print("Server started at port", server_address[1])
    if "open" in sys.argv:
        webbrowser.open_new_tab("http://localhost:8000")
    httpd.serve_forever()

run()
