resofile=$1
resistance=$2

source ~/root_build/bin/thisroot.sh
root << EOF
  TCanvas *canvas = new TCanvas("canvas","Resolution",800,600);

  TTree *t = new TTree("t", "t");
  t->ReadFile("$resofile");

  t->Draw("en:sqrt(pow(sigma_abs/peak_abs,2) - bes*bes*1e-4 - pow(1.92e-7*pow(en, 2.5), 2)*1e-4 ):0:sigma_abs/peak_abs * sqrt( pow(err_sigma_abs/sigma_abs,2)+ pow(err_peak_abs/peak_abs, 2) )");
  auto *ge = new TGraphErrors(t->GetSelectedRows(), t->GetV1(), t->GetV2(), t->GetV3(), t->GetV4());
  ge->SetName("gReso_${resistance}");
  ge->SetTitle("${resistance} #Omega");

  ge->Draw("AP");

  TF1 *f = new TF1("f",
       "sqrt(pow([N(GeV)]/x, 2) + pow([S(%)]/sqrt(x) * 1e-2, 2) + pow([C(%)]*1e-2, 2))",
       0, 200);
  ge->Fit(f);

  ge->GetXaxis()->SetTitle("Beam energy [GeV]");
  ge->GetYaxis()->SetTitle("Energy Resolution (3x3)");

  TLatex l(80, 0.02,
    "#frac{#sigma_{E}}{E} = #frac{N}{E} #oplus #frac{S}{#sqrt{E[GeV]}} #oplus C");
  l.Draw();

  TFile fout("reso_interactive_${resistance}.root", "RECREATE");
  ge->Write("gReso_${resistance}");
  f->Write("fReso");
  canvas->Write("canvas");
  fout.Close();

  canvas->SaveAs("reso_canvas_${resistance}.root");
EOF
