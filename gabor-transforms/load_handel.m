load handel;
v = y'/2;
filename = 'handel.wav';
audiowrite(filename, y, Fs);

plot((1:length(v))/Fs, v);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');
