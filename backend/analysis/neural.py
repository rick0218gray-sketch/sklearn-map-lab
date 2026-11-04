"""CPUで実学習するニューラル推定器。履歴・勾配・内部表現を取得。"""
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split

torch.set_num_threads(2)
class Network(nn.Module):
    def __init__(self, n, out, kind, latent=8, image_shape=None):
        super().__init__(); self.kind=kind; self.n=n; self.image_shape=image_shape
        self.input=nn.Linear(1,16); self.pos=nn.Parameter(torch.randn(1,n,16)*.01)
        self.attn=nn.MultiheadAttention(16,2,batch_first=True)
        self.conv=nn.Conv2d(1,8,3,padding=1) if image_shape else nn.Conv1d(1,8,3,padding=1)
        cell={'rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU}.get(kind,nn.GRU)
        self.cell=cell(1,16,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.head=nn.Linear(8 if kind=='cnn' else 16,out)
        self.weights=None; self.features=None; self.conv_features=None
    def forward(self,x):
        if self.kind=='cnn':
            inputs=x.reshape(-1,1,*self.image_shape) if getattr(self,'image_shape',None) else x[:,None,:]
            z=torch.relu(self.conv(inputs)); self.conv_features=z
            if z.requires_grad: z.retain_grad()
            z=z.mean(tuple(range(2,z.ndim)))
        elif self.kind in ('rnn','lstm','gru'): z=self.cell(x[:,:,None])[0][:,-1,:]
        elif self.kind=='transformer':
            z=self.input(x[:,:,None])+self.pos
            h,w=self.attn(z,z,z,need_weights=True); self.weights=w
            z=torch.relu(z+h).mean(1)
        else: z=self.mlp(x)
        self.features=z
        return self.head(z)

class NeuralBase(BaseEstimator):
    def __init__(self, kind='mlp', epochs=25, lr=.01, random_state=42, image_shape=None):
        self.kind=kind; self.epochs=epochs; self.lr=lr; self.random_state=random_state; self.image_shape=image_shape
    def __setstate__(self,state):
        state.setdefault('image_shape',None);super().__setstate__(state)
    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y)
        self.n_features_in_=X.shape[1]; torch.manual_seed(self.random_state)
        classification=isinstance(self,ClassifierMixin)
        if classification:
            self.classes_,y=np.unique(y,return_inverse=True); out=len(self.classes_)
        else: out=1; y=y.astype(np.float32)
        ids=np.arange(len(y)); strat=y if classification and np.bincount(y).min()>=2 and len(y)*.2>=out else None
        train,val=train_test_split(ids,test_size=.2,random_state=self.random_state,stratify=strat)
        xt=torch.tensor(X[train]); xv=torch.tensor(X[val]); yt=torch.tensor(y[train]); yv=torch.tensor(y[val])
        if not classification: yt=yt.float().reshape(-1,1); yv=yv.float().reshape(-1,1)
        shape=getattr(self,'image_shape',None) if self.kind=='cnn' else None
        if shape and (len(shape)!=2 or min(shape)<2 or np.prod(shape)!=X.shape[1]):raise ValueError('2D CNNの縦×横と変換後特徴数が一致しません。ピクセル列だけを選択してください')
        self.net_=Network(X.shape[1],out,self.kind,image_shape=shape)
        opt=torch.optim.Adam(self.net_.parameters(),lr=self.lr); scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,.98)
        lossfn=nn.CrossEntropyLoss() if classification else nn.MSELoss()
        self.history_={'loss':[],'val_loss':[],'metric':[],'val_metric':[],'lr':[],'grad':[]}
        for epoch in range(self.epochs):
            self.net_.train(); opt.zero_grad(); pred=self.net_(xt); loss=lossfn(pred,yt); loss.backward()
            grad=sum(float(p.grad.square().sum()) for p in self.net_.parameters() if p.grad is not None)**.5
            nn.utils.clip_grad_norm_(self.net_.parameters(),10); opt.step(); self.net_.eval()
            with torch.no_grad():
                pv=self.net_(xv); vl=lossfn(pv,yv)
                mt=float((pred.argmax(1)==yt).float().mean()) if classification else float(torch.sqrt(loss))
                mv=float((pv.argmax(1)==yv).float().mean()) if classification else float(torch.sqrt(vl))
            for k,v in zip(self.history_,[float(loss.detach()),float(vl),mt,mv,opt.param_groups[0]['lr'],grad]): self.history_[k].append(v)
            scheduler.step()
        self.net_.eval(); self.training_X_=X[:64]
        return self
    def _output(self,X):
        with torch.no_grad(): return self.net_(torch.tensor(np.asarray(X,dtype=np.float32))).numpy()
    def predict(self,X):
        p=self._output(X)
        return self.classes_[p.argmax(1)] if hasattr(self,'classes_') else p.ravel()
    def embedding(self,X):
        self._output(X); return self.net_.features.detach().numpy()
    def diagnostics(self,X,target_class=1):
        x=torch.tensor(np.asarray(X,dtype=np.float32),requires_grad=True)
        out=self.net_(x); index=int(target_class) if out.shape[1]>1 else 0
        if not 0<=index<out.shape[1]:raise ValueError('説明対象のクラス番号が範囲外です')
        target=out[:,index].sum(); self.net_.zero_grad(); target.backward()
        result={'saliency':x.grad.detach().numpy(),'activation':self.net_.features.detach().numpy(),
                'weights':np.concatenate([p.detach().numpy().ravel() for p in self.net_.parameters()])}
        if self.net_.weights is not None: result['attention']=self.net_.weights.detach().numpy()
        if self.kind=='cnn':
            f=self.net_.conv_features; g=f.grad.mean(tuple(range(2,f.ndim)),keepdim=True)
            result['gradcam']=torch.relu((g*f).sum(1)).detach().numpy()
        return result
class NeuralClassifier(ClassifierMixin,NeuralBase):
    def predict_proba(self,X):
        p=self._output(X); p=np.exp(p-p.max(1,keepdims=True)); return p/p.sum(1,keepdims=True)
class NeuralRegressor(RegressorMixin,NeuralBase): pass

class ReconstructionNet(nn.Module):
    def __init__(self,n,vae):
        super().__init__(); self.vae=vae
        self.encoder=nn.Sequential(nn.Linear(n,24),nn.ReLU(),nn.Linear(24,8))
        self.mean=nn.Linear(8,4); self.logvar=nn.Linear(8,4)
        self.decoder=nn.Sequential(nn.Linear(4 if vae else 8,24),nn.ReLU(),nn.Linear(24,n))
    def forward(self,x):
        h=self.encoder(x); kl=torch.tensor(0.)
        if self.vae:
            mu,lv=self.mean(h),self.logvar(h).clamp(-8,8)
            z=mu+torch.randn_like(mu)*torch.exp(lv/2) if self.training else mu
            kl=-.5*(1+lv-mu.square()-lv.exp()).mean()
        else: z=h
        self.embedding_=z; self.activation_=h
        return self.decoder(z),kl
class Reconstruction(BaseEstimator):
    def __init__(self,vae=False,epochs=25,contamination=.05,random_state=42):
        self.vae=vae;self.epochs=epochs;self.contamination=contamination;self.random_state=random_state
    def fit(self,X,y=None):
        X=np.asarray(X,dtype=np.float32);self.n_features_in_=X.shape[1];torch.manual_seed(self.random_state)
        tr,va=train_test_split(X,test_size=.2,random_state=self.random_state)
        tr,va=torch.tensor(tr),torch.tensor(va)
        self.net_=ReconstructionNet(X.shape[1],self.vae); opt=torch.optim.Adam(self.net_.parameters(),lr=.01)
        self.history_={'loss':[],'val_loss':[],'metric':[],'val_metric':[],'lr':[],'grad':[]}
        for _ in range(self.epochs):
            self.net_.train();opt.zero_grad();re,kl=self.net_(tr);loss=(re-tr).square().mean()+.01*kl;loss.backward()
            grad=sum(float(p.grad.square().sum()) for p in self.net_.parameters() if p.grad is not None)**.5
            opt.step();self.net_.eval()
            with torch.no_grad(): re,_=self.net_(va);vl=(re-va).square().mean()
            for k,v in zip(self.history_,[float(loss.detach()),float(vl),float(loss.detach()),float(vl),.01,grad]):self.history_[k].append(v)
        self.threshold_=np.quantile(self.errors(X),1-self.contamination); return self
    def reconstruct(self,X):
        with torch.no_grad(): return self.net_(torch.tensor(np.asarray(X,dtype=np.float32)))[0].numpy()
    def errors(self,X): return ((np.asarray(X)-self.reconstruct(X))**2).mean(1)
    def decision_function(self,X): return -self.errors(X)
    def predict(self,X): return np.where(self.errors(X)>self.threshold_,-1,1)
    def embedding(self,X): self.reconstruct(X);return self.net_.embedding_.detach().numpy()
    def diagnostics(self,X):
        x=torch.tensor(np.asarray(X,dtype=np.float32),requires_grad=True);re,_=self.net_(x);(re-x).square().sum().backward()
        return {'saliency':x.grad.detach().numpy(),'activation':self.net_.activation_.detach().numpy(),'weights':np.concatenate([p.detach().numpy().ravel() for p in self.net_.parameters()])}
