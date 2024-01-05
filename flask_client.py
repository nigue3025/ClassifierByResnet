import requests
file_names=['/home/bryanni/BlurryMarkupAll/Blurry/116756.jpg',
            '/home/bryanni/BlurryMarkupAll/Clear/180322.jpg']
resp = requests.post("http://10.35.82.170:5000/predict",
                     files={
                         file_names[0]: open(file_names[0],'rb'),
                         file_names[1]: open(file_names[1], 'rb')
                        })
print(resp.text)