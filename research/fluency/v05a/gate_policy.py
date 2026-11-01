"""v0.5a: pure decision arithmetic; no model execution or file writes."""
from fractions import Fraction as Q


def ballot_gate(ab, ae, be):
    """Each argument maps the identical ten group IDs to two strict item matches."""
    if not (len(ab) == 10 and ab.keys() == ae.keys() == be.keys()):
        raise ValueError('expected the same ten groups')
    for m in (ab, ae, be):
        if any(len(v) != 2 or any(type(x) is not bool for x in v) for v in m.values()):
            raise ValueError('each group requires two boolean item results')
    human = [g for g in sorted(ab) if all(ab[g])]
    shared = [g for g in human if all(ae[g]) and all(be[g])]
    return dict(human_groups=human, common_editor_groups=shared,
                original_pass=len(human) >= 8, permission_for_offline_arms=len(shared) >= 8)


def rate(n, d):
    if type(n) is not int or type(d) is not int or not 0 <= n <= d:
        raise ValueError('invalid count')
    return None if not d else Q(n, d)


def decision(quality, changes, executions, peers, safe, bodies):
    """quality has [alerts, observed_reference, joint_hits] for full & panel24.
    changes/executions: counts on new96, new94, new80. peers: exactly two
    (changes, executions) count pairs, on the same new96 lines. bodies counts
    distinct prespecified panel poems with confirmed B and location OR reason hit.
    safe=[input_replay,negative_control,seven_R_no_extra_B]. Inputs must come
    from a checked gate and keyed receipts, never from unverified human claims.
    """
    if len(quality) != 2 or len(safe) != 3 or len(peers) != 2:
        raise ValueError('incomplete comparison')
    if len(changes) != 3 or len(executions) != 3 or type(bodies) is not int:
        raise ValueError('invalid panels')
    for q, cap in zip(quality, (192, 24)):
        a, r, h = q; rate(a, cap); rate(r, a); rate(h, r)
    if not (0 <= bodies <= quality[1][1] and quality[1][0] <= quality[0][0]
            and quality[1][1] <= quality[0][1] and quality[1][2] <= quality[0][2]):
        raise ValueError('inconsistent nested quality counts')
    if any(s is not None and type(s) is not bool for s in safe):
        raise ValueError('invalid safeguard')
    for c, e, cap in zip(changes, executions, (96,94,80)):
        rate(e,cap); rate(c,e)
    for t in (changes,executions):
        if not 0 <= t[0]-t[1] <= 2 or not 0 <= t[1]-t[2] <= 14:
            raise ValueError('impossible nested masks')
    checks = {'safeguards': False if False in safe else None if None in safe else True}
    for name, (a,r,h) in zip(('precision_all','precision_panel24'), quality):
        checks[name] = None if not a or r != a else rate(h,a) >= Q(4,5)
    checks['five_confirmed_poems'] = None if quality[1][0] < 5 or quality[1][1] != quality[1][0] else bodies >= 5
    checks['absolute_coverage'] = all(n >= m for ns in (changes,executions) for n,m in zip(ns,(10,10,8)))
    for i,(c,e) in enumerate(peers):
        rate(e,96); rate(c,e)
        for j,(own,other) in enumerate(((changes[0],c),(executions[0],e))):
            checks[f'peer_{i}_{j}'] = 'NA_ZERO_CONTROL' if not other else 5*own >= 4*other
    status = 'FAIL' if False in checks.values() else 'INSUFFICIENT' if None in checks.values() else 'REPLICATION_ONLY'
    return {'status':status,'checks':checks,'automatic_filter':False,'deploy':False}
