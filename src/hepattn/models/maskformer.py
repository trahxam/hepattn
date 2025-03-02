from torch import nn
from torch.nn import ModuleList


class MaskFormer(nn.Module):
    def __init__(
        self,
        loss_config: dict,  # noqa: ARG002
        init_nets: ModuleList,
        mask_decoder: nn.Module,
        encoder: nn.Module | None = None,
        pool_net: nn.Module | None = None,
        tasks: nn.ModuleList | None = None,
    ):
        super().__init__()

        self.init_nets = init_nets
        self.encoder = encoder
        self.mask_decoder = mask_decoder
        self.pool_net = pool_net
        if tasks is None:
            tasks = []
        self.tasks = tasks

        # setup loss
        # self.loss = MFLoss(**loss_config, tasks=tasks)  # noqa: ERA001
        self.loss = None

        # check init nets have the same output embedding size
        sizes = {list(init_net.parameters())[-1].shape[0] for init_net in self.init_nets}
        assert len(sizes) == 1

    def forward(self, inputs: dict, mask: dict, labels: dict | None = None):
        # initial embeddings
        embed_x = {}
        for init_net in self.init_nets:
            embed_x[init_net.name] = init_net(inputs)

        assert len(self.init_nets) == 1
        embed_x = next(iter(embed_x.values()))
        input_pad_mask = next(iter(mask.values()))

        # encoder/decoder
        if self.encoder is not None:
            embed_x = self.encoder(embed_x, mask=input_pad_mask)
        preds = {"embed_x": embed_x}

        # get mask decoder outputs
        preds.update(self.mask_decoder(embed_x, input_pad_mask))

        # get the loss, updating the preds and labels with the best matching
        do_loss_matching = True  # set to false for inference timings
        if do_loss_matching:
            preds, labels, loss = self.loss(preds, labels)
        else:
            loss = {}  # disable the bipartite matching
            for task in self.tasks:
                if task.input_type == "queries":
                    preds.update(task(preds, labels))

        # pool the node embeddings
        if self.pool_net is not None:
            preds.update(self.pool_net(preds))

        # configurable tasks here
        for task in self.tasks:
            if task.input_type == "queries":
                task_loss = task.get_loss(preds, labels)
            else:
                task_preds, task_loss = task(preds, labels, input_pad_mask)
                preds.update(task_preds)
            loss.update(task_loss)

        return preds, loss
