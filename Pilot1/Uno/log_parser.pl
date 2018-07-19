#!/usr/bin/perl

while(<>){
  chomp;
  s/[\[\]\,\:]//g; @a=split/\s+/;
  @a = split/\s+/;
  # print join ("\t", @a), "\n";
  push @{ $loss->{$a[4]} }, $a[6];
  push @{ $val_loss->{$a[4]} }, $a[12];
}

print "epoch\tavg_loss\tavg_val_loss\n";
foreach (sort keys(%$loss)){
  # print (join (",", @{$loss->{$_}}), "\n");
  print "Epoch $_:\t", average($loss->{$_}), "\t", average($val_loss->{$_}), "\n"; 
}

sub average {
  my $total;
  $total = $total + $_ foreach @{$_[0]};
  $total / scalar( @{$_[0]} );
}

# print average([1,2,3,4]);
