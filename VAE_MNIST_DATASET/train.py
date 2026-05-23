from model import loss_fn

def train(model,loader,opt,device,epochs=12):
    model.train()

    for e in range(epochs):
        total=0

        for x,_ in loader:
            x=x.view(-1,784).to(device)

            recon,mu,logvar=model(x)
            loss=loss_fn(recon,x,mu,logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {e+1}: {total/len(loader.dataset):.4f}")