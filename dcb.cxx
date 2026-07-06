double DoubleSidedCrystalballFunction(double *x, double *par)
{ 
  double alpha_l = par[0]; 
  double alpha_h = par[1]; 
  double n_l     = par[2]; 
  double n_h     = par[3]; 
  double mean	= par[4]; 
  double sigma	= par[5];
  double N	= par[6];
  float t = (x[0]-mean)/sigma;
  double result;
  double fact1TLessMinosAlphaL = alpha_l/n_l;
  double fact2TLessMinosAlphaL = (n_l/alpha_l) - alpha_l -t;
   double fact1THihgerAlphaH = alpha_h/n_h;
   double fact2THigherAlphaH = (n_h/alpha_h) - alpha_h +t;
   
   if (-alpha_l <= t && alpha_h >= t)
     {
       result = exp(-0.5*t*t);
     }
   else if (t < -alpha_l)
     {
       
       result = exp(-0.5*alpha_l*alpha_l)*pow(fact1TLessMinosAlphaL*fact2TLessMinosAlphaL, -n_l);
       
     }
   else if (t > alpha_h)
     {
       result = exp(-0.5*alpha_h*alpha_h)*pow(fact1THihgerAlphaH*fact2THigherAlphaH, -n_h);
       
     }
   return N*result;
}

void dcb(TH1 *h){
  int firstbin = h->GetXaxis()->GetFirst();
  int lastbin  = h->GetXaxis()->GetLast();
  double xmin = h->GetXaxis()->GetBinLowEdge(firstbin);
  double xmax = h->GetXaxis()->GetBinUpEdge(lastbin);
  auto *f = new TF1("dcb", DoubleSidedCrystalballFunction, xmin, xmax, 7);
  f->SetParameters(2, 2, 2, 2, h->GetMean(), h->GetRMS(), h->Integral());
  f->SetParLimits(0, 0.1, 10);
  f->SetParLimits(1, 0.1, 10);
  f->SetParLimits(2, 1, 10);
  f->SetParLimits(3, 1, 10);
  f->SetParLimits(4, xmin, xmax);
  f->SetParLimits(5, 0, xmax-xmin);
  h->Fit(f, "R");
}
