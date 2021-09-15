import React, {useState} from "react";
import {Card, Container, Row} from "reactstrap";

import Header from "components/Headers/Header.js";
import SongInfo from "components/SongInfo.js";

import SearchBox from "./Sections/SearchBox";


function SearchPage() {

    const [SongList, setSongList] = useState(null);

    return (
        <>
            <Header/>
            <Container className="mt--7" fluid>
                <Row>
                    <div className="col">
                        <Card className="shadow">
                            <SearchBox getSongList={setSongList}/>
                            {
                                SongList ? (<SongInfo song_list={SongList}/>) : null
                            }
                        </Card>
                    </div>
                </Row>
            </Container>
        </>
    );
}

export default SearchPage;
