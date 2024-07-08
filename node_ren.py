import torch
import torch.nn as nn
import torch.nn.functional as F


class NODE_REN(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, alpha = 0.0, linear_output=False):
        """Used by the upper class NODE_REN to guarantee contractivity to the model. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0.
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 0.5           #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        # self.Ptilde = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.D22 = nn.Parameter(torch.randn(ny,nu,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        #Initialization of the last Parameters:
        self.A = torch.zeros(nx,nx,device=device)
        # self.Y= torch.zeros(nx,nx)
        self.D11 = torch.zeros(nq,nq,device=device)
        self.C1 = torch.zeros(nq,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.P = torch.zeros(nx,nx,device=device)
        self.alpha= alpha
        self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        """Used at the end of each batch training for the update of the constrained matrices.
        """
        P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device)
        # P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device) + self.Ptilde - self.Ptilde
        self.P = P
        H = F.linear(self.X,self.X) + self.epsilon*torch.eye(self.nx+self.nq,device=self.device)
        #Partition of H in --> [H1 H2;H3 H4]
        h1,h2 = torch.split(H, (self.nx,self.nq), dim =0) # you split the matrices in two big rows
        H1, H2 = torch.split(h1, (self.nx,self.nq), dim=1) # you split each big row in two chunks
        H3, H4 = torch.split(h2, (self.nx,self.nq), dim=1)

        Y= -0.5*(H1+ self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P),Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda),torch.tril(H4,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda),self.Chi)
        Z = -H2-self.Chi
        self.B1 = F.linear(torch.inverse(P),Z.T)

    def forward(self, xi):
        u = torch.zeros((1, 2))
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1,device=self.device),self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_#,yi

    def output(self, xi, u):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ F.linear(u, self.D12[0,:])  ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By