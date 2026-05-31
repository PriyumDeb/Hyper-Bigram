import torch as tr
import torch.nn.functional as fun
with open("car.txt","r") as f:
    reads=f.read().splitlines()
marked=sorted(list(set(''.join(reads))))
stoi={k:v for v,k in enumerate(marked)}
stoi['<S>']= len(stoi)
stoi['<E>']= len(stoi)
print(len(stoi))
itos={k:v for  v,k in stoi.items()}
inputs=[]
targets=[]
for a in reads:
    clean=['<S>']  + list(a) + ['<E>']
    for c1,c2 in zip(clean,clean[1:]):
        ix=stoi[c1]
        iy=stoi[c2]
        inputs.append(ix)
        targets.append(iy)

inputs=tr.tensor(inputs)
targets=tr.tensor(targets)

xenc= fun.one_hot(inputs,num_classes=len(stoi)).float() 
g=tr.Generator().manual_seed(93648736487)      #Generator not generator
wei=tr.randn((len(stoi),len(stoi)),generator=g,requires_grad=True)
for i in range (50):
    
    logits= xenc @ wei
    mins= logits.exp()
    sums= mins/mins.sum(1,keepdims=True)

    arr=sums[tr.arange(len(inputs)),targets]                #problem arrived
    loss=-1* arr.log().mean()
    wei.grad=None                                          #problem arrived
    loss.backward()
    wei.data+=-40 * wei.grad                               #problem arrived     
    #print(loss.item())         

for i in range (4):
    pre=[]
    io=stoi['<S>']
    while True:
        logits2=wei[io]
        mins2=logits2.exp()     
        sums2=mins2/mins2.sum()
        io=tr.multinomial(sums2,num_samples=1,replacement=True).item()   
        if io==stoi['<E>']:
            break
        pre.append(itos[io])          
    print(''.join(pre))


