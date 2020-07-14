for i in `seq 1 64`;
do
  echo worker $i
  # on cloud:
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python run_evaluation_dopamine.py --base_dir="test_results" --gin_files="test_configs/rainbow.gin" &
  # on macbook for debugging:
  #python extract.py &
  sleep 1.0
done
