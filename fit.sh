enlist=$1
outf=$2
inputdir=$3
resistance=$4



if [ $resistance == 340 ]; then
    scale=3500/150.
elif [ $resistance == 400 ]; then
    scale=1080/40.
elif [ $resistance == 500 ]; then
    scale=3340/100.
else 
    echo "wrong resistance selcted"
    break
fi
dcb_dir=$PWD
echo ${dcb_dir}
echo "en,sigma_abs,zero,err_sigma_abs,peak_abs,err_peak_abs" > $outf



    #FitAmp_3x3_${en}_uncalibrated->GetXaxis()->SetRangeUser(650, 700);
for en in $(cat $enlist);
do
  #root ${inputdir}/FitAmp_3x3_$en.root << EOF
  root ${inputdir}/${en}GeV_reco.root << EOF

    ofstream outf("$outf", std::ios::app);
    .L ${dcb_dir}/dcb.cxx
    h4_reco->Draw("A_tot>>FitAmp_3x3_${en}_uncalibrated(8000,0,8000)","abs(pos_eta-18)<=0.2 && abs(pos_phi-6)<=0.2")
    FitAmp_3x3_${en}_uncalibrated->GetXaxis()->SetRangeUser(${scale} * $en*0.95, ${scale} * $en*1.05);
    FitAmp_3x3_${en}_uncalibrated->Draw()
    FitAmp_3x3_${en}_uncalibrated->GetXaxis()->SetRangeUser(FitAmp_3x3_${en}_uncalibrated->GetMean() - 3*FitAmp_3x3_${en}_uncalibrated->GetRMS(), FitAmp_3x3_${en}_uncalibrated->GetMean() + 3*FitAmp_3x3_${en}_uncalibrated->GetRMS())
    FitAmp_3x3_${en}_uncalibrated->GetXaxis()->SetRangeUser(FitAmp_3x3_${en}_uncalibrated->GetMean() - 3*FitAmp_3x3_${en}_uncalibrated->GetRMS(), FitAmp_3x3_${en}_uncalibrated->GetMean() + 3*FitAmp_3x3_${en}_uncalibrated->GetRMS())
    dcb(FitAmp_3x3_${en}_uncalibrated)
    dcb(FitAmp_3x3_${en}_uncalibrated)
    dcb(FitAmp_3x3_${en}_uncalibrated)
    auto *f = FitAmp_3x3_${en}_uncalibrated->GetFunction("dcb")
    outf << ${en} << "," << f->GetParameter(5) << ",0," << f->GetParError(5) << "," << f->GetParameter(4) << "," << f->GetParError(4) << endl;
    outf.close()
    FitAmp_3x3_${en}_uncalibrated->SaveAs("${inputdir}/FitAmp_3x3_${en}_fitted.root");
    .q
EOF
done


