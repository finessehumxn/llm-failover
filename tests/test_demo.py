from llm_failover import demo


async def test_demo_runs_offline_and_shows_each_scenario(capsys):
    await demo.main()
    out = capsys.readouterr().out
    assert "answered by 'secondary'; fell_back=True" in out
    assert "CONFIG   primary" in out
    assert "OPENED" in out
    assert "skipped  primary" in out
    assert "RequestRejected from 'primary'; secondary was called 0 times" in out
    assert "1 configuration error(s)" in out
