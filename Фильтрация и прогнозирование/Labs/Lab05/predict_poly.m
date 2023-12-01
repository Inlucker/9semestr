function [Xsint,poly_pred] = predict_poly(MJDsc,X,MJD_pred, deg)
%predicts polynomial of degree deg
%INPUT
%MJDsc initial dates
%X  initial signal
%MJD_pred  dates to predict
%deg - degree of polynomial deg=2 - 1 order - line
%OUTPUT
%Xsint  time series after polynomial model remooved
%poly_pred - predictions according to polynomial model

  p_coef=polyfit(MJDsc,X,deg);
  model_poly = transpose(polyval(p_coef,MJDsc));
  Xsint=X-model_poly';
  poly_pred = polyval(p_coef,MJD_pred);
  plot(MJDsc,X,'--',MJDsc,model_poly,'-',MJD_pred,poly_pred)
end

