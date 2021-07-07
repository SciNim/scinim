import scinim / experimental

let x = @[1, 2, 3, 4, 5]
let y = linspaceT(0.0, 10.0, 20)
let yInt = arangeT(1, 6, 1)

echo max(x)
echo max(y)

for i, x in y:
  echo i
  echo x

echo cumSum(x)
echo $cumSum(y)

#echo cumProd(x)
#echo cumProd(y)
#
#echo cumCount(x, 1)
#echo cumCount(yInt, 1)
#
#echo product(x)
#echo product(yInt)
#
#echo sumSquares(x)
#echo sumSquares(yInt)
#
#echo argmax(x)
#echo argmax(yInt)
#
#echo argmin(x)
#echo argmin(yInt)
#
#echo histogram(x)
