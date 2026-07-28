import tensorly as tl
import numpy as np

class LinearFactorModel:
    def __init__(self, k, params, iters, tol = 1e-4, reg = None, random_state = 0, print_iter = 10, check_convergence = 10, verbose = True, normalize_activities = True, b_factor_type = "ntf", learn_factors = None):
        self.k = k
        self.S = tl.zeros((k, k, k))
        for i in range(k):
            self.S[i, i, i] = 1
        self.params = params
        self.iters = iters
        self.reg = reg
        self.random_state = random_state
        self.print_iter = print_iter
        self.check_convergence = check_convergence
        self.verbose = verbose
        self.normalize_activities = normalize_activities
        self.b_factor_type = b_factor_type
        self.factors = None
        self.activities = None
        self.learn_factors = learn_factors
        self.tol = tol
    def loss_terms(self):
        return {"fit_Xb" : np.linalg.norm(self.Xs["Xb"] - tl.tenalg.multi_mode_dot(self.S, [self.activities, self.factors["Vb_0"], self.factors["Vb_1"]]))**2 / 2 if self.b_factor_type == "ntf"\
                        else np.linalg.norm(self.Xs["Xb"] - self.activities @ self.factors["Vb"].T)**2 / 2,
                "fit_Xe" : np.linalg.norm(self.Xs["Xe"] - self.activities @ self.factors["Ve"].T)**2 / 2,
                "fit_Xm" : np.linalg.norm(self.Xs["Xm"] - self.activities @ self.factors["Vm"].T)**2 / 2,
                "U_l1" : np.sum(np.abs(self.activities)), 
                "U_l2" : np.linalg.norm(self.activities)**2 / 2, 
                "U_lap" : np.trace(self.activities.T @ (self.reg["D_lap"] - self.reg["A_lap"]) @ self.activities)/2}
    def loss(self):
        l = self.loss_terms()
        return (self.params['lamda_b']*l['fit_Xb'] + self.params['lamda_e']*l['fit_Xe'] + self.params['lamda_m']*l['fit_Xm'] +\
                self.params['mu_1']*l['U_l1'] + self.params['mu_2']*l['U_l2'] + self.params['mu_lap']*l['U_lap'])
    def fit(self, Xs):
        self.Xs = Xs
        Xb, Xe, Xm = self.Xs["Xb"], self.Xs["Xe"], self.Xs["Xm"]
        self.trace = []
        lamda_b, lamda_e, lamda_m = self.params["lamda_b"], self.params["lamda_e"], self.params["lamda_m"]
        mu_1, mu_2, mu_lap = self.params["mu_1"], self.params["mu_2"], self.params["mu_lap"]
        D_lap, A_lap = self.reg["D_lap"], self.reg["A_lap"]
        # init 
        np.random.seed(self.random_state)
        n = Xb.shape[0]
        if self.activities is None:
            self.activities = np.random.rand(n, self.k)
        if self.b_factor_type == "ntf":
            if self.factors is None:
                self.factors = {
                    "Ve" : np.random.rand(self.k, Xe.shape[1]).T,
                    "Vm" : np.random.rand(self.k, Xm.shape[1]).T, 
                    "Vb_0" : np.random.rand(Xb.shape[1], self.k), 
                    "Vb_1" : np.random.rand(Xb.shape[2], self.k)
                }
            Vb0, Vb1, Ve, Vm = self.factors["Vb_0"], self.factors["Vb_1"], self.factors["Ve"], self.factors["Vm"]
        elif self.b_factor_type == "nmf":
            if self.factors is None:
                self.factors = {
                    "Ve" : np.random.rand(self.k, Xe.shape[1]).T,
                    "Vm" : np.random.rand(self.k, Xm.shape[1]).T, 
                    "Vb" : np.random.rand(self.k, Xb.shape[1]).T 
                }
            Vb, Ve, Vm = self.factors["Vb"], self.factors["Ve"], self.factors["Vm"]
        if self.learn_factors is None:
            self.learn_factors = list(self.factors.keys())
        # copy local variables
        U = self.activities
        # iters
        obj = 1e9
        for i in range(self.iters):
            if self.b_factor_type == "ntf":
                # update activities
                B0 = tl.unfold(tl.tenalg.multi_mode_dot(self.S, [Vb0, Vb1], [1, 2]), 0)
                U *= (lamda_b*tl.unfold(Xb, 0) @ B0.T + lamda_e*(Xe @ Ve) + lamda_m*(Xm @ Vm) + mu_lap*(A_lap @ U)) / \
                                    (U @ (lamda_b*B0 @ B0.T + lamda_e * Ve.T @ Ve + lamda_m * Vm.T @ Vm) + mu_1 + mu_2*U + mu_lap*(D_lap @ U))
                if self.normalize_activities:
                    U /= U.sum(0)
                # update Vb0, Vb1
                if "Vb_0" in self.learn_factors:
                    B1 = tl.unfold(tl.tenalg.multi_mode_dot(self.S, [U, Vb1], [0, 2]), 1)
                    Vb0 *= (tl.unfold(Xb, 1) @ B1.T) / (Vb0 @ B1 @ B1.T)
                if "Vb_1" in self.learn_factors:
                    B2 = tl.unfold(tl.tenalg.multi_mode_dot(self.S, [U, Vb0], [0, 1]), 2)
                    Vb1 *= (tl.unfold(Xb, 2) @ B2.T) / (Vb1 @ B2 @ B2.T)
                if "Ve" in self.learn_factors:
                    Ve *= Xe.T @ U / (Ve @ U.T @ U)
                if "Vm" in self.learn_factors:
                    Vm *= Xm.T @ U / (Vm @ U.T @ U)
            elif self.b_factor_type == "nmf":
                U *= (lamda_b*(Xb @ Vb) + lamda_e*(Xe @ Ve) + lamda_m*(Xm @ Vm) + mu_lap*(A_lap @ U)) /\
                    (U @ (lamda_b * Vb.T @ Vb + lamda_e * Ve.T @ Ve + lamda_m * Vm.T @ Vm) + mu_1 + mu_2*U + mu_lap*(D_lap @ U))
                if self.normalize_activities:
                    U /= U.sum(0)
                Vb *= (Xb.T @ U / (Vb @ U.T @ U))
                Ve *= (Xe.T @ U / (Ve @ U.T @ U))
                Vm *= (Xm.T @ U / (Vm @ U.T @ U))
            if (i % self.print_iter == 0) or (i % self.check_convergence == 0):
                obj_new = self.loss()
                self.trace += [obj_new, ]
                if self.verbose and (i % self.print_iter == 0):
                    print(f"Iteration {i},\t self.loss = {obj_new}")
                isconverged = np.abs(obj - obj_new)/obj < self.tol
                obj = obj_new
                if isconverged:
                    break
        return self.trace
    def reconstruction(self):
        if self.b_factor_type == "ntf":
            return {"Xb" : tl.tenalg.multi_mode_dot(self.S,[self.activities, self.factors["Vb_0"], self.factors["Vb_1"]]),
                            "Xe" : self.activities @ self.factors["Ve"].T, 
                            "Xm" : self.activities @ self.factors["Vm"].T}
        elif self.b_factor_type == "nmf":
            return {"Xb" : self.activities @ self.factors["Vb"].T,
                            "Xe" : self.activities @ self.factors["Ve"].T, 
                            "Xm" : self.activities @ self.factors["Vm"].T}
    def predict_single(self, X, what = "expr"):
        lamda_b, lamda_e, lamda_m = self.params["lamda_b"], self.params["lamda_e"], self.params["lamda_m"]
        mu_1, mu_2, mu_lap = self.params["mu_1"], self.params["mu_2"], self.params["mu_lap"]
        activities_pred = np.random.rand(X.shape[0], self.k)
        if (what == "expr") or (what == "marker"):
            factor = {"expr" : "Ve", "marker" : "Vm"}[what]
            V = self.factors[factor]
            lamda = {"expr" : lamda_e, "marker" : lamda_m}[what]
            for i in range(self.iters):
                activities_pred *= (lamda * X @ V) / (lamda * activities_pred @ V.T @ V + mu_1 + mu_2 * activities_pred)
                if self.normalize_activities:
                    activities_pred /= activities_pred.sum(0)
                if i % self.print_iter == 0:
                    print(f"Iteration {i},\t loss = {np.linalg.norm(X - activities_pred @ V.T)**2}")
        elif what == "barcode":
            Vb0, Vb1 = self.factors["Vb_0"], self.factors["Vb_1"]
            B0 = tl.unfold(tl.tenalg.multi_mode_dot(self.S, [Vb0, Vb1], [1, 2]), 0)
            for i in range(self.iters):
                activities_pred *= (lamda_b*tl.unfold(X, 0) @ B0.T) / \
                                    (activities_pred @ (lamda_b*B0 @ B0.T) + mu_1 + mu_2 * activities_pred)
                if self.normalize_activities:
                    activities_pred /= activities_pred.sum(0)
                if i % self.print_iter == 0:
                    loss = np.linalg.norm(X - tl.tenalg.multi_mode_dot(self.S, [activities_pred, self.factors["Vb_0"], self.factors["Vb_1"]]))**2
                    print(f"Iteration {i},\t loss = {loss}")
        return activities_pred
    def predict_joint(self, Xe, Xm):
        lamda_b, lamda_e, lamda_m = self.params["lamda_b"], self.params["lamda_e"], self.params["lamda_m"]
        mu_1, mu_2, mu_lap = self.params["mu_1"], self.params["mu_2"], self.params["mu_lap"]
        activities_pred = np.random.rand(Xe.shape[0], self.k)
        Ve = self.factors["Ve"]
        Vm = self.factors["Vm"]
        for i in range(self.iters):
            activities_pred *= (lamda_e*(Xe @ Ve) + lamda_m*(Xm @ Vm)) / (activities_pred @ (lamda_e * Ve.T @ Ve + lamda_m * Vm.T @ Vm))
            if i % self.print_iter == 0:
                print(np.linalg.norm(Xe - activities_pred @ Ve.T)**2, np.linalg.norm(Xm - activities_pred @ Vm.T)**2)
        return activities_pred
        
    
def fit_basis(X, U, maxiter = 100):
    V = np.random.rand(X.shape[1], U.shape[1])
    for i in range(maxiter):
        V *= X.T @ U / (V @ U.T @ U)
    return V
