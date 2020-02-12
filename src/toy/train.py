import torch, tqdm
import torch.utils.data

import src

def main(dataset, testset, model, lr):

    dset = torch.utils.data.TensorDataset(*dataset)
    load = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)

    lossf = torch.nn.L1Loss()

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-6)

    epochs = 300
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    bar = tqdm.tqdm(range(epochs), ncols=80)

    avg = src.util.MovingAverage(momentum=0.99)
    done = False
    for epoch in bar:

        if done:
            continue

        model.train()
        for X, Y in load:
            Yh = model(X).mean(dim=-1)
            loss = lossf(Yh, Y)
            done = torch.isnan(loss)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg.update(loss.item())
            bar.set_description("Loss %.5f" % avg.peek())

            if done:
                break

        sched.step()

    lossf = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        X_data, Y_data = dataset
        Yh_data = model(X_data).mean(dim=-1)
        data_loss = lossf(Yh_data, Y_data).item()**0.5
            
        X_test, Y_test = testset
        Yh_test = model(X_test).mean(dim=-1)
        test_loss = lossf(Yh_test, Y_test).item()**0.5

        return data_loss, test_loss
