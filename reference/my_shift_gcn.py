from Shift_GCN.model.shift_gcn import *
class MyModel(nn.Module):
    def __init__(self, num_class=64, num_point=16, in_channels=2, onlyFeature=False):
        super(MyModel, self).__init__()
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64,num_point, residual=False)
        self.l2 = TCN_GCN_unit(64, 64,num_point)
        self.l3 = TCN_GCN_unit(64, 64,num_point)
        self.l4 = TCN_GCN_unit(64, 64,num_point)
        self.l5 = TCN_GCN_unit(64, 128,num_point)
        self.l6 = TCN_GCN_unit(128, 128,num_point)
        self.l7 = TCN_GCN_unit(128, 128,num_point)
        self.l8 = TCN_GCN_unit(128, 256,num_point)
        self.l9 = TCN_GCN_unit(256, 256,num_point)
        self.l10 = TCN_GCN_unit(256, 256,num_point)
        self.onlyFeature=onlyFeature
        if onlyFeature:
            self.fc=nn.Linear(256, in_channels)
        else:
            self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        # bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        # batch，xyz，time, joints, person number

        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x=x.permute(0,4,1,2,3).reshape(N*M,C,T,V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        # c_new = x.size(1)
        # # x = x.reshape(N, M, c_new, T,V).permute(0,3,1,2,4)
        # # x=x.reshape(N,T,M,-1)
        # # x = x.mean(3)#.mean(1)
        # # N,T,M,C
        # return self.fc(x)
        # return x
        c_new = x.size(1)
        x = x.view(N, M, c_new, T,V)
        if self.onlyFeature:
            return self.fc(x.permute(0,1,3,4,2)).permute(0,1,4,2,3)
        else:
            x = x.mean(4)
            x=x.permute(0,3,1,2)
            return self.fc(x)