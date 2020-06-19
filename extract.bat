FOR /L %%A IN (1,1,3) DO (
  ECHO %%A
START "" /B python extract.py
)