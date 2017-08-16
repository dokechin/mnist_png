use strict;
use warnings;

open my $train, ">./training.ctl";
open my $test, ">./testing.ctl";
my @png = glob "./*/*/*.png";
for my $png(@png){
  $png =~ /(training|testing)\/(\d)\/(\d+)\.png/;
  my $file;
  if ($1 eq "training" ){
      $file = $train;
  }
  else{
      $file = $test;
  }
  printf $file "%s\t%s\t./%s/%s/%s.png\n", $3 , $2, $1 , $2, $3;
}

close $train;
close $test;

