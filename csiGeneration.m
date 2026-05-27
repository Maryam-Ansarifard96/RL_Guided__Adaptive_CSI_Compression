clc; clear; close all;
rng(12345);

%% ------------------------ Simulation Parameters -------------------------
K = 2;                         % try 2 / 4 / 6 / 8
NumBSant = 64;
TxArraySize = [8 4 2 1 1];
numSlots = 8000;
slotDuration = 0.5e-3;
timeVec = (0:numSlots-1) * slotDuration;

carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30;
carrier.NSizeGrid = 52;
waveInfo = nrOFDMInfo(carrier);
Nsub = carrier.NSizeGrid * 12;
SampleRate = waveInfo.SampleRate;

c = 3e8; Fc = 3.5e9;
PL0_dB = 30; d0 = 1; PLE = 3.2;

TxPower_dBm = 23;
NoiseFigure_dB = 7;
ThermalNoise_dBmHz = -174;
BW = SampleRate;

%% ------------------------ K-Aware Tuning -------------------------------
if K <= 2
    transWidth = 140;
    minDwell = 500;  maxDwell = 1100;
    initSwitchLo = 400; initSwitchHi = 900;

    stateSpeedMean = [1.0, 7.0, 24.0];
    stateSpeedJitter = [0.2, 1.5, 4.0];
    statePenalty_dB = [1.0, 3.5, 7.0];
    stateShadowStd_dB = [0.8, 2.5, 4.5];

    scen1_mix = 0.90; scen1_ripple = 0.001; scen1_rank = 3;
    scen2_mix = 0.20; scen2_ripple = 0.10;  scen2_rank = 12;
    scen3_ripple = 0.55; scen3_hard = 0.95; scen3_scramble = 0.22;

elseif K <= 4
    transWidth = 120;
    minDwell = 450;  maxDwell = 1200;
    initSwitchLo = 350; initSwitchHi = 950;

    stateSpeedMean = [1.0, 7.5, 24.0];
    stateSpeedJitter = [0.2, 1.6, 4.2];
    statePenalty_dB = [1.0, 2.5, 3.5];
    stateShadowStd_dB = [0.8, 2.0, 2.8];

    scen1_mix = 0.88; scen1_ripple = 0.0015; scen1_rank = 4;
    scen2_mix = 0.22; scen2_ripple = 0.10;   scen2_rank = 12;
    scen3_ripple = 0.56; scen3_hard = 0.98;  scen3_scramble = 0.23;

elseif K <= 6
    transWidth = 110;
    minDwell = 420;  maxDwell = 1300;
    initSwitchLo = 320; initSwitchHi = 1000;

    stateSpeedMean = [1.0, 8.0, 25.0];
    stateSpeedJitter = [0.2, 1.7, 4.3];
    statePenalty_dB = [1.0, 3.0, 4.0];
    stateShadowStd_dB = [0.9, 2.2, 3.0];

    scen1_mix = 0.88; scen1_ripple = 0.0015; scen1_rank = 4;
    scen2_mix = 0.20; scen2_ripple = 0.105;  scen2_rank = 12;
    scen3_ripple = 0.58; scen3_hard = 1.00;  scen3_scramble = 0.24;

else
    transWidth = 95;
    minDwell = 380;  maxDwell = 1250;
    initSwitchLo = 260; initSwitchHi = 1000;

    stateSpeedMean = [1.0, 8.0, 26.0];
    stateSpeedJitter = [0.2, 1.8, 4.2];
    statePenalty_dB = [1.0, 3.0, 4.2];
    stateShadowStd_dB = [0.9, 2.2, 3.1];

    scen1_mix = 0.88; scen1_ripple = 0.0015; scen1_rank = 4;
    scen2_mix = 0.22; scen2_ripple = 0.11;   scen2_rank = 13;
    scen3_ripple = 0.60; scen3_hard = 1.02;  scen3_scramble = 0.25;
end

%% ------------------------ Allocate Tensors ------------------------------
CSI = complex(zeros(Nsub, 1, NumBSant, K, numSlots));
userSNR_dB = zeros(K, numSlots);
scenarioLabels = zeros(K, numSlots);
userSpeed = zeros(K, numSlots);


slotBudgetBits = zeros(1, numSlots);
L_DELAY_FOR_PYTHON = 64;
fullDimPython = 2 * NumBSant * L_DELAY_FOR_PYTHON;
LATENT_Q_BITS = 10;
latentBits = @(cr) LATENT_Q_BITS * max(8, round(fullDimPython * cr));

% Scenario-specific best CR target:
% Scenario 1 -> 1/32, Scenario 2 -> 1/16, Scenario 3 -> 1/8
preferredCR = [1/32, 1/16, 1/8];

%% ------------------------ Global Scenario Schedule ----------------------
globalState = 1;
globalTargetState = 1;
globalTransitioning = false;
globalTransCounter = 0;
globalStateSeq = [1 2 3 1 3 2 1 3 1 2 3 1 3 2 1];

globalDwellSeq = [1100 450 1100 ...
                  1100 1100 450 ...
                  1100 1100 1100 ...
                  450 1100 1100 ...
                  1100 450 1100];
seqIdx = 1;
globalNextSwitch = initSwitchLo;

%% ------------------------ Initialize Users ------------------------------
users = struct();
for u = 1:K
    users(u).pos = [30 + 15*u + 10*randn(), 20*randn()];

    theta0 = 2*pi*rand();
    users(u).velocity = stateSpeedMean(1) * [cos(theta0), sin(theta0)];

    users(u).hardnessScale = 0.85 + 0.4*rand();
    users(u).pathBias_dB = -0.5 + 2.5*rand();
    users(u).rippleBias = 0.9 + 0.25*rand();
    users(u).rankBias = 0.9 + 0.2*rand();

    users(u).ch1 = createChannel('CDL-A', TxArraySize, Fc, SampleRate, 5e-9);
    users(u).ch2 = createChannel('CDL-C', TxArraySize, Fc, SampleRate, 180e-9);
    users(u).ch3 = createChannel('CDL-E', TxArraySize, Fc, SampleRate, 1100e-9);

    users(u).state = globalState;
    users(u).targetState = globalTargetState;
    users(u).isTransitioning = false;
    users(u).transCounter = 0;
    users(u).nextSwitch = inf;
end

fprintf('Starting simulation: %d slots, K=%d, transWidth=%d\n', numSlots, K, transWidth);

%% ------------------------ Main Time Loop --------------------------------
for t = 1:numSlots
    if mod(t,500) == 0
        fprintf('Processing Slot %d/%d...\n', t, numSlots);
    end

    % -------- synchronized global scenario switching --------
    if t == globalNextSwitch
        globalTransitioning = true;
        globalTransCounter = 0;

        oldState = globalState;

        seqIdx = seqIdx + 1;
        if seqIdx > numel(globalStateSeq)
            seqIdx = 1;
        end

        globalTargetState = globalStateSeq(seqIdx);

        if globalTargetState == oldState
            candidateStates = setdiff(1:3, oldState);
            globalTargetState = candidateStates(1);
        end

        dwell = globalDwellSeq(seqIdx);
        globalNextSwitch = min(numSlots + 1, t + dwell);

        fprintf('[Slot %d] GLOBAL scenario: %d -> %d\n', ...
            t, oldState, globalTargetState);
    end

    if globalTransitioning
        globalTransCounter = globalTransCounter + 1;
        wGlobal = min(1, globalTransCounter / transWidth);
    else
        wGlobal = 0;
    end

    s0Global = globalState;
    s1Global = globalTargetState;

    effectiveGlobalState = s0Global;
    if globalTransitioning && wGlobal > 0.5
        effectiveGlobalState = s1Global;
    end

    % This budget makes the best fixed CR scenario-dependent.
    slotBudgetBits(t) = K * latentBits(preferredCR(effectiveGlobalState));

    for u = 1:K
        users(u).state = s0Global;
        users(u).targetState = s1Global;
        users(u).isTransitioning = globalTransitioning;
        users(u).transCounter = globalTransCounter;

        w = wGlobal;
        s0 = s0Global;
        s1 = s1Global;

        speed0 = stateSpeedMean(s0) + stateSpeedJitter(s0) * randn();
        speed1 = stateSpeedMean(s1) + stateSpeedJitter(s1) * randn();
        speed_mps = max(0.5, (1-w)*speed0 + w*speed1);
        userSpeed(u,t) = speed_mps;

        theta = 0.12*sin(2*pi*t/1200 + 0.6*u) + 0.08*cos(2*pi*t/1800 + u);
        users(u).velocity = speed_mps * [cos(theta), sin(theta)];
        users(u).pos = users(u).pos + users(u).velocity * slotDuration;

        newDoppler = (speed_mps / c) * Fc;

        chList = {users(u).ch1, users(u).ch2, users(u).ch3};
        for ci = 1:3
            if chList{ci}.MaximumDopplerShift ~= newDoppler
                release(chList{ci});
                chList{ci}.MaximumDopplerShift = newDoppler;
            end
        end
        users(u).ch1 = chList{1};
        users(u).ch2 = chList{2};
        users(u).ch3 = chList{3};

        [g1, s1info] = users(u).ch1();
        [g2, s2info] = users(u).ch2();
        [g3, s3info] = users(u).ch3();

        f1 = getPathFilters(users(u).ch1);
        f2 = getPathFilters(users(u).ch2);
        f3 = getPathFilters(users(u).ch3);

        H1_full = nrPerfectChannelEstimate(carrier, g1, f1, 0, s1info);
        H2_full = nrPerfectChannelEstimate(carrier, g2, f2, 0, s2info);
        H3_full = nrPerfectChannelEstimate(carrier, g3, f3, 0, s3info);

        H1 = squeeze(H1_full(:,1,:,:));
        H2 = squeeze(H2_full(:,1,:,:));
        H3 = squeeze(H3_full(:,1,:,:));

        H_from = pickStateChannel(s0, H1, H2, H3);
        H_to   = pickStateChannel(s1, H1, H2, H3);
        Hsym = (1-w)*H_from + w*H_to;

        effectiveState = s0;
        if globalTransitioning && w > 0.5
            effectiveState = s1;
        end

        hs = users(u).hardnessScale;
        rb = users(u).rippleBias;
        rk = users(u).rankBias;

        if effectiveState == 1
            Hsym = applyAntennaMixing(Hsym, scen1_mix);
            Hsym = applyFrequencyRipple(Hsym, scen1_ripple * rb, t, u);
            Hsym = applyLowRankProjection(Hsym, max(2, round(scen1_rank * rk)));

        elseif effectiveState == 2
            Hsym = applyAntennaMixing(Hsym, scen2_mix);
            Hsym = applyFrequencyRipple(Hsym, scen2_ripple * hs * rb, t, u);
            Hsym = applyLowRankProjection(Hsym, max(4, round(scen2_rank * rk)));

        else
            Hsym = applyFrequencyRipple(Hsym, scen3_ripple * hs * rb, t, u);
            Hsym = applyHardStateDistortion(Hsym, scen3_hard * hs, t, u);
            Hsym = applySubcarrierScramble(Hsym, scen3_scramble * hs, t, u);
        end

        dist = norm(users(u).pos);
        localShadow = randn() * stateShadowStd_dB(effectiveState);

        PL_dB = PL0_dB ...
            + 10*PLE*log10(max(dist,d0)/d0) ...
            + localShadow ...
            + users(u).pathBias_dB;

        pen0 = statePenalty_dB(s0);
        pen1 = statePenalty_dB(s1);
        scenarioPenalty_dB = (1-w)*pen0 + w*pen1;
        PL_dB = PL_dB + scenarioPenalty_dB;

        noiseFloor_dBm = ThermalNoise_dBmHz + 10*log10(BW) + NoiseFigure_dB;
        RxPower_dBm = TxPower_dBm - PL_dB;
        userSNR_dB(u,t) = RxPower_dBm - noiseFloor_dBm;

        CSI(:, 1, :, u, t) = Hsym * 10^(-PL_dB/20);
        scenarioLabels(u,t) = effectiveState;
    end

    if globalTransitioning && globalTransCounter >= transWidth
        globalState = globalTargetState;
        globalTransitioning = false;
    end
end

%% ------------------------ Save ------------------------------------------
filename = sprintf('CSI_GlobalScenario_CRTarget_K%d_v2.mat', K);
save(filename, ...
    'CSI', ...
    'userSNR_dB', ...
    'scenarioLabels', ...
    'userSpeed', ...
    'slotBudgetBits', ...
    'preferredCR', ...
    '-v7.3');

fprintf('Saved to %s\n', filename);

%% ------------------------ Helpers ---------------------------------------
function chan = createChannel(profileType, arraySize, fc, rate, delaySpread)
    chan = nrCDLChannel;
    chan.DelayProfile = profileType;
    chan.CarrierFrequency = fc;
    chan.SampleRate = rate;
    chan.DelaySpread = delaySpread;
    chan.TransmitAntennaArray.Size = arraySize;
    chan.ReceiveAntennaArray.Size = [1 1 1 1 1];
    chan.ChannelFiltering = false;
end

function H = pickStateChannel(state, H1, H2, H3)
    if state == 1
        H = H1;
    elseif state == 2
        H = H2;
    else
        H = H3;
    end
end

function Hout = applyFrequencyRipple(Hin, amp, t, userIdx)
    [Nsub, Nant] = size(Hin);
    f = (0:Nsub-1)' / max(1, Nsub-1);
    ripple = 1 + amp * sin(2*pi*(3*f + 0.0008*t + 0.4*userIdx));
    Hout = Hin .* repmat(ripple, 1, Nant);
end

function Hout = applyAntennaMixing(Hin, amp)
    [~, Nant] = size(Hin);
    v = linspace(-1,1,Nant);
    kernel = exp(-((v(:)-v(:)').^2)/0.20);
    kernel = kernel ./ max(sum(kernel,2), 1e-12);
    Hsmooth = Hin * kernel;
    Hout = (1-amp)*Hin + amp*Hsmooth;
end

function Hout = applyHardStateDistortion(Hin, strength, t, userIdx)
    [Nsub, Nant] = size(Hin);
    f = (0:Nsub-1)' / max(1, Nsub-1);

    ripple = 1 + strength * ( ...
        0.6*sin(2*pi*(5*f + 0.0010*t + 0.3*userIdx)) + ...
        0.4*sin(2*pi*(11*f + 0.0017*t + 0.5*userIdx)) );

    notchCenter = 0.2 + 0.6*abs(sin(0.0009*t + userIdx));
    notchWidth = 0.03;
    notch = 1 - 0.55 * exp(-((f - notchCenter).^2) / (2*notchWidth^2));

    antPhase = exp(1j * strength * pi * linspace(-1,1,Nant));
    antPhase = repmat(antPhase, Nsub, 1);

    Hout = Hin .* repmat(ripple .* notch, 1, Nant);
    Hout = Hout .* antPhase;
end

function Hout = applyLowRankProjection(Hin, rankKeep)
    [U,S,V] = svd(Hin, 'econ');
    r = min([rankKeep, size(S,1), size(S,2)]);
    Hout = U(:,1:r) * S(1:r,1:r) * V(:,1:r)';
end

function Hout = applySubcarrierScramble(Hin, amp, t, userIdx)
    [Nsub, Nant] = size(Hin);
    rng(1000 + userIdx + floor(t/25));
    phaseJitter = exp(1j * amp * 2*pi * randn(Nsub, Nant));
    Hout = Hin .* phaseJitter;
end
