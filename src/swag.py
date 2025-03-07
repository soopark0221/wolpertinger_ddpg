class SWAG(torch.nn.Module):
    def __init__(self, base, var_clamp=1e-6):
        super(SWAG, self).__init__()

        self.base_model = base
        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())
        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'
        self.mean = torch.zeros(self.num_parameters)
        self.sq_mean = torch.zeros(self.num_parameters)
        self.n_models = torch.zeros(1, dtype=torch.long)
        self.max_rank = 50
        self.cpv_mat_sqrt = torch.empty(0, self.num_parameters, dtype=torch.float32)
        self.rank = torch.zeros(1, dtype=torch.long)
    def collect_model(self, base_model):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        w = flatten([param.detach().cpu() for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))

        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        dev_vector = w - self.mean
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5

    def set_swa(self):
        set_weights(self.base_model, self.mean, self.model_device)

    def sample(self, target_model, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()
        #self.cov_factor = self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5
        eps_low_rank = torch.randn(self.cov_factor.size()[0])
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance.sqrt() * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        self.set_weights(target_model, sample, self.model_device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()

    def flatten(self, lst):
        tmp = [i.contiguous().view(-1,1) for i in lst]
        return torch.cat(tmp).view(-1)

    def set_weights(self, model, vector, device=None):
        offset = 0
        for param in model.parameters():
            param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
            offset += param.numel()

