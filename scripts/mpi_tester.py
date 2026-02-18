"""
Self-contained script for running MPI tests using 'Pytest'.

This file was extracted from the 'runtests' project
(https://github.com/bccp/runtests) by merging files

    . run-mpitests.py
    . runtests/tester.py
    . runtests/mpi/tester.py

and stripping them of any non-essential parts.

NOTE: This is still work-in-progress!

"""
import pytest
import traceback
import warnings
import sys
import os
import contextlib
import shutil
import time
import re

from argparse   import ArgumentParser
from contextlib import contextmanager
from io         import StringIO

#===============================================================================
# UTILITIES
#===============================================================================
def _make_clean_dir(path):
    print("Purging %s ..." % path)
    try:
        shutil.rmtree(path)
    except OSError:
        pass
    try:
        os.makedirs(path)
    except OSError:
        pass

#-------------------------------------------------------------------------------
def fix_titles(s):
    pattern = '=====+'
    return re.sub(pattern, lambda x: x.group(0).replace('=', '-'), s)

#-------------------------------------------------------------------------------
class Rotator(object):
    """ in a rotator every range runs in terms """
    def __init__(self, comm):
        self.comm = comm
    def __enter__(self):
        self.comm.Barrier()
        for i in range(self.comm.rank):
            self.comm.Barrier()
    def __exit__(self, type, value, tb):
        for i in range(self.comm.rank, self.comm.size):
            self.comm.Barrier()
        self.comm.Barrier()

#-------------------------------------------------------------------------------
@contextmanager
def nompi(comm):
    errored = False
    error = None
    try:
        yield
    except Exception as e:
        errored = True
        error = e
    finally:
        anyerrored = any(comm.allgather(errored))

    if anyerrored:
        if error is None:
            raise RuntimeError("Some ranks failed")
        else:
            raise error

#===============================================================================
# MPI TEST DECORATOR
#===============================================================================
def MPITest(commsize):
    """
    A decorator that repeatedly calls the wrapped function,
    with communicators of varying sizes.

    This converts the test to a generator test; therefore the
    underlyig test shall not be a generator test.

    Parameters
    ----------
    commsize: scalar or tuple
        Sizes of communicator to use

    Usage
    -----
    @MPITest(commsize=[1, 2, 3])
    def test_stuff(comm):
        pass
    """
    from mpi4py import MPI
    if not isinstance(commsize, (tuple, list)):
        commsize = (commsize,)

    sizes = sorted(list(commsize))

    def dec(func):

        @pytest.mark.parametrize("size", sizes)
        def wrapped(size, *args):
            if MPI.COMM_WORLD.size < size:
                pytest.skip("Test skipped because world is too small. Include the test with mpirun -n %d" % (size))

            color = 0 if MPI.COMM_WORLD.rank < size else 1
            comm = MPI.COMM_WORLD.Split(color)
            try:
                if color == 0:
                    rt = func(*args, comm=comm)
                if color == 1:
                    rt = None
                    #pytest.skip("rank %d not needed for comm of size %d" %(MPI.COMM_WORLD.rank, size))
            finally:
                comm.Free()
                MPI.COMM_WORLD.barrier()

            return rt
        wrapped.__name__ = func.__name__
        return wrapped
    return dec

#===============================================================================
# MAIN TESTER CLASS
#===============================================================================
class Tester( object ):
    """
    Run MPI-enabled tests using pytest, building the project first.

    Examples::
        $ python runtests.py my/module
        $ python runtests.py --single my/module
        $ python runtests.py my/module/tests/test_abc.py
        $ python runtests.py --mpirun="mpirun -np 4" my/module
        $ python runtests.py --mpirun="mpirun -np 4"
    """

    @staticmethod
    def pytest_addoption(parser):
        """
        Add command-line options to specify MPI and coverage configuration
        """
        parser.addoption("--mpirun", default="mpirun -n 4",
                help="Select MPI launcher, e.g. mpirun -n 4")

        parser.addoption("--single", default=False, action='store_true',
                help="Do not run via MPI launcher. ")

        parser.addoption("--mpisub", action="store_true", default=False,
                help="run process as a mpisub")

        parser.addoption("--mpisub-site-dir", default=None, help="site-dir in mpisub")

    #---------------------------------------------------------------------------
    @staticmethod
    def pytest_collection_modifyitems(session, config, items):
        """
        Modify the ordering of tests, such that the ordering will be
        well-defined across all ranks running
        """

        # sort the tests
        items[:] = sorted(items, key=lambda x: str(x))

    #---------------------------------------------------------------------------
    def __init__(self):

        self.ROOT_DIR = os.path.abspath( os.path.curdir )
        self.TEST_DIR = os.path.join( self.ROOT_DIR, '__test__' )

    #---------------------------------------------------------------------------
    @property
    def comm(self):
        from mpi4py import MPI
        return MPI.COMM_WORLD

    #---------------------------------------------------------------------------
    def main(self, argv):
        # must bail after first dead test; avoiding a fault MPI collective state.
        argv.insert(1, '-x')

        config = self._get_pytest_config(argv)
        args = config.known_args_namespace

        # print help and exit
        if args.help:
            return config.hook.pytest_cmdline_main(config=config)

        # import project from system path
        args.pyargs = True

        # build / setup on the master
        if not args.mpisub:

            self._initialize_dirs(args)
            site_dir = None
            if not args.single:
                self._launch_mpisub(args, site_dir)

        else:

            capman = config.pluginmanager.getplugin('capturemanager')
            if capman:
                if hasattr(capman, 'suspend_global_capture'):
                    capman.suspend_global_capture()
                else:
                    capman.suspendcapture()

            # test on mpisub.
            if args.mpisub_site_dir:
                site_dir = args.mpisub_site_dir
                sys.path.insert(0, site_dir)
                os.environ['PYTHONPATH'] = site_dir

                # if we are here, we will run the tests, either as sub or single
                # fix the path of the modules we are testing
                config.args = self._fix_test_paths(site_dir, config.args)

        if args.mpisub:
            self._begin_capture()

        # run the tests
        try:
            code = None
            with self._run_from_testdir(args):
                code = config.hook.pytest_cmdline_main( config=config )

        except:
            if args.mpisub:
                self._sleep()
                self.oldstderr.write("Fatal Error on Rank %d\n" % self.comm.rank)
                self.oldstderr.write(traceback.format_exc())
                self.oldstderr.flush()
                self.comm.Abort(-1)
            else:
                traceback.print_exc()
                sys.exit(1)

        if args.mpisub:
            self._end_capture_and_exit(code)
        else:
            sys.exit(code)

    #---------------------------------------------------------------------------
    def _launch_mpisub(self, args, site_dir):

        # extract the mpirun run argument
        parser = ArgumentParser(add_help=False)
        # these values are ignored. This is a hack to filter out unused argv.
        parser.add_argument("--single", default=False, action='store_true')
        parser.add_argument("--mpirun", default=None)
        _args, additional = parser.parse_known_args()

        # now call with mpirun
        mpirun = args.mpirun.split()
        cmdargs = [sys.executable, sys.argv[0], '--mpisub']

        if site_dir is not None:
            # mpi subs will use system version of package
            cmdargs.extend(['--mpisub-site-dir=' + site_dir])

        # workaround the strict openmpi oversubscribe policy
        # the parameter is found from
        # https://github.com/open-mpi/ompi/blob/ba47f738871ff06b8e8f34b8e18282b9fe479586/orte/mca/rmaps/base/rmaps_base_frame.c#L169
        # see the faq:
        #   https://www.open-mpi.org/faq/?category=running#oversubscribing
        os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'

        os.execvp(mpirun[0], mpirun + cmdargs + additional)

        # if we are here os.execvp has failed; bail
        sys.exit(1)

    #---------------------------------------------------------------------------
    def _sleep(self):
        time.sleep(0.04 * self.comm.rank)

    #---------------------------------------------------------------------------
    def _begin_capture(self):
        self.oldstdout = sys.stdout
        self.oldstderr = sys.stderr
        self.newstdout = StringIO()
        self.newstderr = StringIO()

        if self.comm.rank != 0:
            sys.stdout = self.newstdout
            sys.stderr = self.newstderr

    #---------------------------------------------------------------------------
    def _end_capture_and_exit(self, code):
        if code != 0:
            # if any rank has a failure, print the error and abort the world.
            self._sleep()
            if self.comm.rank != 0:
                self.oldstderr.write("Test Failure due to rank %d\n" % self.comm.rank)
                self.oldstderr.write(self.newstdout.getvalue())
                self.oldstderr.write(self.newstderr.getvalue())
                self.oldstderr.flush()
            self.comm.Abort(-1)

        self.comm.barrier()
        with Rotator(self.comm):
            if self.comm.rank != 0:
                self.oldstderr.write("\n")
                self.oldstderr.write("=" * 32 + " Rank %d / %d " % (self.comm.rank, self.comm.size) + "=" * 32)
                self.oldstderr.write("\n")
                self.oldstderr.write(fix_titles(self.newstdout.getvalue()))
                self.oldstderr.write(fix_titles(self.newstderr.getvalue()))
                self.oldstderr.flush()

        sys.exit(0)

    #---------------------------------------------------------------------------
    @contextlib.contextmanager
    def _run_from_testdir(self, args):
        if not args.mpisub:
            with super(Tester, self)._run_from_testdir(args):
                yield
                return
        cwd = os.getcwd()

        try:
            assert(os.path.exists(self.TEST_DIR))
            self.comm.barrier()
            os.chdir(self.TEST_DIR)
            yield
        finally:
            os.chdir(cwd)

    #---------------------------------------------------------------------------
    def _get_pytest_config(self, argv):
        """
        Return the ``pytest`` configuration object based on the
        command-line arguments
        """
        import _pytest.config as _config

        plugins = [self]

        # disable pytest-cov
        argv += ['-p', 'no:pytest_cov']

        # Do not load any conftests initially.
        # This is a hack to avoid ConftestImportFailure raised when the
        # pytest does its initial import of conftests in the source directory.
        # This prevents any manipulation of the command-line via conftests
        argv += ['--noconftest']

        # get the pytest configuration object
        try:
            config = _config._prepareconfig(argv, plugins)
        except _config.ConftestImportFailure as e:
            tw = _config.py.io.TerminalWriter(sys.stderr)
            for line in traceback.format_exception(*e.excinfo):
                tw.line(line.rstrip(), red=True)
            tw.line("ERROR: could not load %s\n" % (e.path), red=True)
            raise

        # Restore the loading of conftests, which was disabled earlier
        # Conftest files will now be loaded properly at test time
        config.pluginmanager._noconftest = False

        return config

    #---------------------------------------------------------------------------
    def _initialize_dirs(self, args):
        """
        Initialize the ``build/test/`` directory
        """
        _make_clean_dir(self.TEST_DIR)

    #---------------------------------------------------------------------------
    def _fix_test_paths(self, site_dir, args):
        """
        Fix the paths of tests to run to point to the corresponding
        tests in the site directory
        """
        def fix_test_path(x):
            p = x.split('::')
            p[0] = os.path.relpath(os.path.abspath(p[0]), self.ROOT_DIR)
            p[0] = os.path.join(site_dir, p[0])
            return '::'.join(p)
        return [fix_test_path(x) for x in args]

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":

    tester = Tester()
    tester.main( sys.argv[1:] )
