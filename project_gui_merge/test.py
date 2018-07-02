import torch

# a = torch.ones(1,128)
# file = open('./face_re_data/face_features.pt','wb')
#
# name = []
# name.append('wug')
# # name.append('yvans')
# torch.save((name,a),file)
# file.close()
name,face_features = torch.load('face_re_data/face_features.pt')
# name = name[1:]
# face_features = face_features[1:]
# file = open('./face_re_data/face_features.pt','wb')
#
# torch.save((name,face_features),file)
# file.close()
# # print((face_features[1]).size(0))
print(name)
print(face_features.size())

# a = torch.cat((a,face_features),0)
# file = open('./face_re_data/face_features.pt','wb')
# torch.save((name,a),file)
# file.close()
# name,face_features = torch.load('./face_re_data/face_features.pt')
# print(name)

