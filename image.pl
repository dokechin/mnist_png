use strict;
use warnings;
use AI::MXNet qw('mx');
use Data::Dumper;
use Test::More tests => 1;

 my $testing_ite = mx->io->ImageRecordIter(
 {  batch_size => 1, data_shape=> [1,28,28],label_width =>1, path_imgrec => "testing.rec", path_root => '.' });


 my $training_ite = mx->io->ImageRecordIter(
 {  batch_size => 1, data_shape=> [1,28,28],label_width =>1, path_imgrec => "training.rec", path_root => '.' });


# for my $data (@{$training_ite}){
#   print Dumper($data);
#   print $data->data->[0]->aspdl;
#   print $data->label->[0]->aspdl;
# }

# Create a place holder variable for the input data
my $data = mx->symbol->Variable('data');

my $conv1= mx->symbol->Convolution(data => $data, name => 'conv1', num_filter => 32, kernel => [3,3], stride => [2,2]);
my $bn1  = mx->symbol->BatchNorm(data => $conv1, name => "bn1");
my $act1 = mx->symbol->Activation(data => $bn1, name => 'act1', act_type => "relu");
my $mp1  = mx->symbol->Pooling(data => $act1, name => 'mp1', kernel => [2,2], stride =>[2,2], pool_type=>'max');
my $conv2= mx->symbol->Convolution(data => $mp1, name => 'conv2', num_filter => 50, kernel=>[3,3], stride=>[2,2]);
my $bn2  = mx->symbol->BatchNorm(data => $conv2, name=>"bn2");
my $act2 = mx->symbol->Activation(data => $bn2, name=>'act2', act_type=>"relu");
my $mp2  = mx->symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

my $fl1   = mx->symbol->Flatten(data => $mp2, name=>"fl1");
my $fc1  = mx->symbol->FullyConnected(data => $fl1,  name=>"fc1", num_hidden=>100);
my $act3 = mx->symbol->Activation(data => $fc1, name=>'act3', act_type=>"relu");
my $fc2  = mx->symbol->FullyConnected(data => $act3, name=>'fc2', num_hidden=>30);
my $act4 = mx->symbol->Activation(data => $fc2, name=>'act4', act_type=>"relu");
my $fc3  = mx->symbol->FullyConnected(data => $act4, name=>'fc3', num_hidden=>10);
my $softmax = mx->symbol->SoftmaxOutput(data => $fc3, name => 'softmax');

my $model = mx->mod->Module(
    symbol => $softmax,       # network structure
);
$model->fit(
   $training_ite,
   eval_data => $testing_ite,
   optimizer_params=>{learning_rate=>0.01, momentum=> 0.9},
   num_epoch=>2
);
my $res = $model->score($testing_ite, mx->metric->create('acc'));
ok($res->{accuracy} > 0.8);